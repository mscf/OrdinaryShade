"""Source provenance and generated-code mappings."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A one-based span in a Python shader source file."""

    path: str
    line: int
    column: int = 1
    end_line: int | None = None
    end_column: int | None = None

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.column}"


@dataclass(frozen=True, slots=True)
class SourceMapEntry:
    """Relate generated shader lines to their originating Python function."""

    generated_start_line: int
    generated_end_line: int
    source: SourceSpan
    symbol: str


def function_span(candidate) -> SourceSpan | None:
    function = getattr(candidate, "function", candidate)
    try:
        lines, start = inspect.getsourcelines(function)
        path = inspect.getsourcefile(function) or inspect.getfile(function)
    except (OSError, TypeError):
        return None
    return SourceSpan(path, start, 1, start + len(lines) - 1, None)


def module_source_map(source: str, candidate) -> tuple[SourceMapEntry, ...]:
    span = function_span(candidate)
    if span is None:
        return ()
    return (
        SourceMapEntry(1, max(1, source.count("\n") + 1), span,
                       getattr(candidate, "__name__", span.path)),
    )


_RELATIVE_LINE = re.compile(r"\bline (\d+)\b")


def annotate_error(error: Exception, candidate) -> Exception:
    """Return an equivalent diagnostic with absolute Python provenance."""
    span = function_span(candidate)
    if span is None or getattr(error, "source_span", None) is not None:
        return error
    message = str(error)
    match = _RELATIVE_LINE.search(message)
    line = span.line
    if match:
        # The parsed source is dedented and starts at the decorated function.
        line = span.line + int(match.group(1)) - 1
    absolute = SourceSpan(span.path, line)
    error.args = (f"{absolute.format()}: {message}",)
    error.source_span = absolute
    return error
