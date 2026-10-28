"""Fetch a bounded, file-level PyPI metadata corpus; never install packages."""
from __future__ import annotations

import email
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RELEASES = {
    'requests': ['2.31.0', '2.32.0', '2.32.3', '2.32.5'],
    'urllib3': ['1.26.18', '2.0.0', '2.0.7', '2.2.3', '2.5.0'],
    'idna': ['3.4', '3.7', '3.10'],
    'charset-normalizer': ['3.1.0', '3.3.2', '3.4.2'],
    'certifi': ['2023.7.22', '2024.8.30', '2025.1.31'],
    'httpx': ['0.24.1', '0.27.2', '0.28.1'],
    'httpcore': ['0.17.3', '1.0.5', '1.0.9'],
    'anyio': ['3.7.1', '4.4.0', '4.9.0'],
    'sniffio': ['1.3.0', '1.3.1'],
    'h11': ['0.14.0', '0.16.0'],
    'typing-extensions': ['4.8.0', '4.12.2', '4.14.0'],
    'exceptiongroup': ['1.1.3', '1.2.2', '1.3.0'],
}

def request(url, *, accept=None):
    if urlparse(url).scheme != 'https' or urlparse(url).hostname not in {'pypi.org', 'files.pythonhosted.org'}:
        raise ValueError('Unexpected upstream URL')
    headers = {'User-Agent': 'DeltaWeave-metadata-research/0.1'}
    if accept:
        headers['Accept'] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return response.read(), dict(response.headers)

def main():
    from packaging.utils import parse_wheel_filename, canonicalize_name
    records, failures, sources = [], [], []
    cache = ROOT / '.cache' / 'pypi'
    cache.mkdir(parents=True, exist_ok=True)
    observed = datetime.now(timezone.utc).isoformat()
    for project, versions in RELEASES.items():
        url = f'https://pypi.org/simple/{project}/'
        raw, headers = request(url, accept='application/vnd.pypi.simple.v1+json')
        index = json.loads(raw)
        (cache / f'{project}-index.json').write_bytes(raw)
        sources.append({'project': project, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'serial': headers.get('x-pypi-last-serial', index.get('meta', {}).get('_last-serial'))})
        for version in versions:
            candidates = []
            for entry in index['files']:
                if not entry['filename'].endswith('.whl'):
                    continue
                name, parsed_version, _, tags = parse_wheel_filename(entry['filename'])
                if canonicalize_name(name) != canonicalize_name(project) or str(parsed_version) != version:
                    continue
                if not any(t.abi == 'none' and t.platform == 'any' and t.interpreter in {'py3', 'py2'} for t in tags):
                    continue
                meta = entry.get('core-metadata', entry.get('dist-info-metadata'))
                if isinstance(meta, dict) and meta.get('sha256'):
                    candidates.append(entry)
            if not candidates:
                failures.append({'project': project, 'version': version, 'reason': 'No hash-addressed universal-wheel metadata available'})
                continue
            entry = sorted(candidates, key=lambda f: f['filename'])[0]
            metadata_url = entry['url'].split('#')[0] + '.metadata'
            metadata, _ = request(metadata_url)
            expected = entry.get('core-metadata', entry.get('dist-info-metadata'))['sha256']
            if hashlib.sha256(metadata).hexdigest() != expected:
                raise RuntimeError(f'Metadata hash mismatch: {entry["filename"]}')
            (cache / (entry['filename'] + '.metadata')).write_bytes(metadata)
            parsed = email.message_from_bytes(metadata)
            if canonicalize_name(parsed['Name']) != canonicalize_name(project) or parsed['Version'] != version:
                raise RuntimeError('Metadata identity mismatch')
            records.append({'project': canonicalize_name(project), 'version': version,
                            'filename': entry['filename'], 'metadata_url': metadata_url,
                            'metadata_sha256': expected, 'wheel_sha256': entry['hashes']['sha256'],
                            'requires_python': parsed.get('Requires-Python'),
                            'requires_dist': parsed.get_all('Requires-Dist', []),
                            'provides_extra': parsed.get_all('Provides-Extra', []),
                            'yanked': bool(entry.get('yanked', False)),
                            'yanked_reason': entry.get('yanked') if isinstance(entry.get('yanked'), str) else None,
                            'upload_time': entry.get('upload-time'), 'observed_at': observed})
            print(f'Verified {project}=={version}: {entry["filename"]}', flush=True)
    destination = ROOT / 'data' / 'public'
    destination.mkdir(parents=True, exist_ok=True)
    result = {'observed_at': observed,
              'scope': 'Selected file-level universal-wheel metadata; bounded candidate universe, not all PyPI; declared constraints only, not runtime compatibility.',
              'yank_history': 'Current observations only; no historical yank time inferred.',
              'sources': sources, 'records': records, 'missing': failures}
    (destination / 'pypi-catalog.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'records': len(records), 'missing': len(failures), 'path': str(destination / 'pypi-catalog.json')}), flush=True)

if __name__ == '__main__':
    main()
