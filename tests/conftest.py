"""Make the analysis modules in code/ importable for the tests (the pipeline imports them as top-level modules)."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
