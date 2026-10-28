"""Build deterministic local state-control fixtures, with split manifests."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deltaweave.learning_data import write_datasets

if __name__ == "__main__":
    print(json.dumps(write_datasets(ROOT / "data" / "learning"), indent=2))
