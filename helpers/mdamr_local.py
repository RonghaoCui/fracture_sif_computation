import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MDAMR_SRC = (
    PROJECT_ROOT
    / "external"
    / "mdamr-main"
    / "src"
)

if str(MDAMR_SRC) not in sys.path:
    sys.path.insert(0, str(MDAMR_SRC))