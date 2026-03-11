import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def resources_path(*parts: str) -> Path:
    return repo_path("resources", *parts)


def models_path(*parts: str) -> Path:
    return repo_path("models", *parts)


def output_path(*parts: str) -> Path:
    return repo_path("output", *parts)
