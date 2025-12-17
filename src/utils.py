from pathlib import Path

_current_file = Path(__file__).resolve()


ROOT_DIR = _current_file.parent.parent

DATA_DIR = ROOT_DIR / "data"
CRD3_DATA_DIR = DATA_DIR / "CRD3"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# add dirs as needeed

if __name__ == "__main__":
    print(f"Project Root is: {ROOT_DIR}")
    print(f"Data Dir is: {DATA_DIR}")