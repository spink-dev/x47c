import os
import mmap

def read(path: str, binary: bool = False):
    mode = "rb" if binary else "r"
    with open(path, mode, encoding=None if binary else "utf-8") as f:
        return f.read()

def write(path: str, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
    with open(path, mode, encoding=None if isinstance(data, (bytes, bytearray)) else "utf-8") as f:
        f.write(data)

def mmap_read(path: str):
    f = open(path, "rb")
    return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def exists(path: str) -> bool:
    return os.path.exists(path)

def size(path: str) -> int:
    return os.path.getsize(path) if exists(path) else 0
