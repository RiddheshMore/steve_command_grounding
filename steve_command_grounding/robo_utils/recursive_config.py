from __future__ import annotations
import datetime
import os
import yaml

class Config:
    def __init__(self, file=None):
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        if file is None:
            file = "config.yaml"
        elif not file.endswith(".yaml"):
            file = f"{file}.yaml"

        def load_recursive(config_name: str, stack: list[str]) -> dict:
            if config_name in stack:
                raise AssertionError("Attempting to build recursive configuration.")

            # Look for configs folder relative to this file
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", config_name)
            
            if not os.path.exists(config_path):
                print(f"Warning: Config file {config_path} not found.")
                return {}

            with open(config_path, "r", encoding="UTF-8") as file_handle:
                cfg = yaml.safe_load(file_handle)

            base = (
                {}
                if "extends" not in cfg
                else load_recursive(cfg["extends"], stack + [config_name])
            )
            base = _recursive_update(base, cfg)
            return base

        self._config = load_recursive(file, [])
        
        if "project_root_dir" not in self._config:
            # Default to workspace root if possible
            self._config["project_root_dir"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def get_subpath(self, subpath: str) -> str:
        subpath_dict = self._config.get("subpaths", {})
        if subpath not in subpath_dict:
            raise ValueError(f"Subpath {subpath} not known.")
        base_path = os.path.normpath(self._config.get("project_root_dir", ""))
        if os.path.isabs(subpath):
            return os.path.normpath(subpath)
        path_ending = os.path.normpath(subpath_dict[subpath])
        return os.path.join(base_path, path_ending)

    def __getitem__(self, item):
        return self._config.get(item)

    def __setitem__(self, key, value):
        self._config[key] = value

    def get(self, key, default=None):
        return self._config.get(key, default)

    def get_config(self):
        return self._config

def _recursive_update(base: dict, cfg: dict) -> dict:
    for k, v in cfg.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base
