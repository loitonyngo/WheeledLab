import yaml
from pathlib import Path
from functools import lru_cache
import os

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "f1tenth" / "config" / "f1tenth_config.yaml"

def get_config_path() -> Path:
    """
    Returns the config path.
    Priority:
    1. Environment variable WHEELEDLAB_CONFIG
    2. Default config path
    """
    env_path = os.getenv("WHEELEDLAB_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


@lru_cache(maxsize=1)  # cache so we don’t reload yaml each time
def load_config() -> dict:
    """
    Loads and caches the config dictionary.
    """
    path = get_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
