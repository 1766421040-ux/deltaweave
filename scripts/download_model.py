"""Download the pinned official DeepSeek checkpoint; no paid service calls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
REVISION = 'ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'
FILES = ['config.json', 'generation_config.json', 'model.safetensors',
         'tokenizer.json', 'tokenizer_config.json', 'LICENSE', 'README.md']

def main():
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'hf'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    from huggingface_hub import HfApi, snapshot_download
    output = ROOT / '.cache' / 'deepseek-r1-distill-qwen-1.5b'
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError('Unexpected source revision')
    meta = {f.rfilename: f for f in info.siblings}
    if set(FILES) - set(meta):
        raise RuntimeError('Required upstream files missing')
    print(f'Downloading official {MODEL} at {REVISION}', flush=True)
    snapshot_download(MODEL, revision=REVISION, allow_patterns=FILES,
                      local_dir=output, max_workers=2)
    manifest = {'model': MODEL, 'revision': REVISION, 'files': {}}
    for name in FILES:
        path = output / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        sha = digest.hexdigest()
        lfs = meta[name].lfs
        if lfs and (sha != lfs.sha256 or path.stat().st_size != lfs.size):
            raise RuntimeError(f'Upstream LFS checksum mismatch: {name}')
        manifest['files'][name] = {'sha256': sha, 'bytes': path.stat().st_size,
                                   'upstream_blob_id': meta[name].blob_id}
    (output / 'download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'download_verified': True, 'path': str(output),
                      'bytes': sum(v['bytes'] for v in manifest['files'].values())}), flush=True)

if __name__ == '__main__':
    main()
