from pathlib import Path
import sys


def ensure_src_path() -> None:
    src_path = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src_path))
