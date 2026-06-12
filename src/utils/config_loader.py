from pathlib import Path
import yaml


class ConfigNode(dict):
    

    def __getattr__(self, item):
        value = self.get(item)

        if isinstance(value, dict):
            return ConfigNode(value)

        return value


class ConfigLoader:

    @staticmethod
    def load(config_path: str):

        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        return ConfigNode(config)       