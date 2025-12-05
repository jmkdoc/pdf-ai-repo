import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML files"""
    config = {}
    
    # Load main settings
    settings_path = Path(__file__).parent / "settings.yaml"
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            config.update(yaml.safe_load(f))
    
    # Load Gemini config
    gemini_path = Path(__file__).parent / "gemini_config.yaml"
    if gemini_path.exists():
        with open(gemini_path, 'r') as f:
            config.update(yaml.safe_load(f))
    
    # Replace environment variables
    def replace_env_vars(obj):
        if isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        return obj
    
    config = replace_env_vars(config)
    
    return config

# Global config instance
config = load_config()
