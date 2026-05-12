# radio/config.py
import os
from pathlib import Path

_ENV_ROOT = "RADIO_ROOT_DIR"
_ROOT_DIR: Path | None = None

def _resolve_root(path: str | os.PathLike | None = None) -> Path:
    """set ROOT_DIR:
    1) set_root_dir()
    2) RADIO_ROOT_DIR
    3) Path.cwd()
    """
    if path is not None: return Path(path).expanduser().resolve()

    env = os.getenv(_ENV_ROOT)
    if env: return Path(env).expanduser().resolve()

    return Path.cwd().resolve()

def set_root_dir(path: str | os.PathLike) -> Path:
    global _ROOT_DIR
    _ROOT_DIR = _resolve_root(path)
    return _ROOT_DIR

def get_root_dir() -> Path:
    global _ROOT_DIR
    if _ROOT_DIR is None: _ROOT_DIR = _resolve_root()
    return _ROOT_DIR

def get_data_dir() -> Path:
    data_dir = get_root_dir() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_log_dir() -> Path:
    log_dir = get_data_dir() / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir