"""Optional validation helpers for generated backend source."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

from ..errors import CompilerUnavailableError, ShaderCompilationError


def validate_wgsl(source: str, *, validator: str = "naga") -> None:
    """Parse and semantically validate WGSL source with Naga.

    Naga is a development-time tool rather than an Ordinary Shade runtime
    dependency. Install ``naga-cli`` to make this function available.
    """
    executable = shutil.which(validator)
    if executable is None and validator == "naga":
        rustup_naga = Path.home() / ".cargo" / "bin" / "naga"
        if rustup_naga.is_file():
            executable = str(rustup_naga)
    if executable is None:
        raise CompilerUnavailableError(
            f"could not locate {validator!r}; install naga-cli to validate WGSL"
        )
    with tempfile.TemporaryDirectory(prefix="ordinaryshade-wgsl-") as directory:
        source_path = Path(directory) / "shader.wgsl"
        source_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            (executable, str(source_path)),
            capture_output=True,
            text=True,
        )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ShaderCompilationError(detail or "Naga rejected generated WGSL")
