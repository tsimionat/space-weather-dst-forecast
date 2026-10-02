import sys
from pathlib import Path

# make "import config" and "import src..." work when running pytest from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
