from pathlib import Path, PurePath
from typing import Any, List, Union

import yaml

PathLike = Union[Path, PurePath, str]


def glob(folder: PurePath, pattern: str) -> List[PurePath]:
    files = Path(folder).glob(pattern=pattern)
    files = sorted(list(files))
    files = [PurePath(f) for f in files]
    return files


def read_yaml(path: PathLike) -> Any:
    assert Path(path).exists() and Path(path).is_file()
    with open(PurePath(path), "r") as f:
        y = yaml.safe_load(f)
    return y


def check_file(path: PathLike) -> bool:
    return Path(path).exists() and Path(path).is_file()


def check_folder(path: PathLike) -> bool:
    return Path(path).exists() and Path(path).is_dir()


def fix_ext(ext: str) -> str:
    return "." + ext.lstrip(".")
