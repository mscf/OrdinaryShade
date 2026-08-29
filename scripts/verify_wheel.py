import sys
from pathlib import Path
import zipfile


wheel = Path(sys.argv[1])
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
required = {
    "ordinaryshade/__init__.py",
    "ordinaryshade/compiler.py",
    "ordinaryshade/ir.py",
    "ordinaryshade/backends/glsl.py",
}
missing = required - names
if missing:
    raise SystemExit(f"wheel is missing: {sorted(missing)}")
print(f"Verified {wheel}: {len(names)} packaged files")

