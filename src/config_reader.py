from pathlib import Path

import yaml

from src.logger import get_logger

logger = get_logger(__name__)


def read_config(config_path):
    """Read and parse a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file. Can be either a string or Path object.

    Returns
    -------
    dict
        Parsed configuration as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified config file does not exist.
    yaml.YAMLError
        If there's an error parsing the YAML file.

    Examples
    --------
    >>> config = read_config('config/config.yaml')
    >>> print(config['data_ingestion']['bucket_name'])
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise


if __name__ == "__main__":
    config = read_config("config/config.yaml")
    print(config["data_ingestion"]["bucket_name"])
