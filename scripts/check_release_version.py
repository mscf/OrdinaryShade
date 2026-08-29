import sys
import tomllib
from pathlib import Path


tag = sys.argv[1] if len(sys.argv) == 2 else ""
version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
expected = f"v{version}"
if tag != expected:
    raise SystemExit(f"release tag {tag!r} does not match {expected!r}")
print(f"Release tag {tag} matches ordinaryshade {version}")

