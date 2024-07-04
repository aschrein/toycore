import numpy as np
from pathlib import Path
import sys
import os

def mkdir_recursive(p: Path):
    if not p.exists():
        p.mkdir(parents=True)

def get_or_create_tmp(name=".tmp"):
    p = Path(name)
    if not p.exists():
        mkdir_recursive(p)
    return p.absolute()

def find_folder_with_file(name, root="."):
    p = Path(root).absolute()
    while os.path.exists(p) and p.parent != p:
        # print(f"Checking {p / name}")
        if os.path.exists(p / name):
            # print(f"Found {p / name}")
            return p / name
        p = p.parent
    return None
