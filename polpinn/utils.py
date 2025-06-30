import configparser
from os.path import expanduser
from pathlib import Path

import polpinn


CONFIG_FILE = Path(expanduser("~")) / ".config" / (__package__ + ".conf")
PROJECT_DIR = Path(polpinn.__file__).parent.parent.parent


def get_data_dir():
    if not CONFIG_FILE.exists():
        generate_config()
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return Path(config.get(section="DATA", option="path"))


def get_output_dir():
    if not CONFIG_FILE.exists():
        generate_config()
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return Path(config.get(section="OUTPUT", option="path"))


def generate_config():
    """Generate the config file in the home folder."""

    config = configparser.ConfigParser(allow_no_value=True)

    config.add_section("DATA")
    config.set("DATA", "# path to the root directory for data.")
    data_dir = PROJECT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    config.set("DATA", "path", str(data_dir))

    config.add_section("OUTPUT")
    config.set("OUTPUT", "# path to the root directory to write outputs.")
    output_dir = PROJECT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    config.set("OUTPUT", "path", str(output_dir))

    CONFIG_FILE.parent.mkdir(exist_ok=True)
    with open(str(CONFIG_FILE), "w") as configfile:
        config.write(configfile)
    print(f"Please edit and update config file {CONFIG_FILE}.")
