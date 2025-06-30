from pathlib import Path
from polpinn.utils import CONFIG_FILE, get_data_dir, get_output_dir, generate_config


def test_generate_config():
    generate_config()
    assert Path(CONFIG_FILE).exists()


def test_get_data_dir():
    generate_config()
    d = get_data_dir()
    assert isinstance(d, Path)


def test_get_output_dir():
    generate_config()
    d = get_output_dir()
    assert isinstance(d, Path)
