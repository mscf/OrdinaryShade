import sys
from pathlib import Path
import zipfile


wheel = Path(sys.argv[1])
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
required = {
    "ordinaryshade/__init__.py",
    "ordinaryshade/compiler/__init__.py",
    "ordinaryshade/compiler/compiler.py",
    "ordinaryshade/ir/__init__.py",
    "ordinaryshade/targets/__init__.py",
    "ordinaryshade/targets/glsl.py",
    "ordinaryshade/targets/wgsl.py",
}
missing = required - names
if missing:
    raise SystemExit(f"wheel is missing: {sorted(missing)}")
forbidden = {
    "ordinaryshade/compiler.py",
    "ordinaryshade/diagnostics.py",
    "ordinaryshade/entrypoints.py",
    "ordinaryshade/errors.py",
    "ordinaryshade/ir.py",
    "ordinaryshade/lowering.py",
    "ordinaryshade/reflection.py",
    "ordinaryshade/types.py",
    "ordinaryshade/validation.py",
}
unexpected = forbidden & names
unexpected.update(name for name in names if name.startswith("ordinaryshade/backends/"))
if unexpected:
    raise SystemExit(f"wheel contains obsolete package paths: {sorted(unexpected)}")
print(f"Verified {wheel}: {len(names)} packaged files")
