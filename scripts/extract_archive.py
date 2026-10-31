"""Unpack a zip whose file names were stored in CP866 (Windows Russian archives) without path traversal.

Usage: uv run python scripts/extract_archive.py ARCHIVE.zip OUT_DIR
"""

import sys
import zipfile
from pathlib import Path


def main(src: str, dst: str) -> None:
    out = Path(dst).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as archive:
        for info in archive.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                try:
                    name = name.encode("cp437").decode("cp866")
                except UnicodeError:
                    pass
            target = (out / name).resolve()
            if not str(target).startswith(str(out) + "/"):
                print(f"skip unsafe path: {name}")
                continue
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
            print(f"{info.file_size:>9}  {name}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
