"""Shader entry-point decorators."""

from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType

from .errors import ShaderTypeError


@dataclass(frozen=True, slots=True)
class ComputeShader:
    function: FunctionType
    workgroup_size: tuple[int, int, int]
    capabilities: tuple[str, ...] = ()

    @property
    def __name__(self):
        return self.function.__name__


@dataclass(frozen=True, slots=True)
class ShaderFunction:
    """A reusable typed shader function without an execution stage."""

    function: FunctionType

    @property
    def __name__(self):
        return self.function.__name__


@dataclass(frozen=True, slots=True)
class ExternalFunction:
    """A typed function supplied by the embedding shader ABI."""
    function: FunctionType

    @property
    def __name__(self):
        return self.function.__name__


@dataclass(frozen=True, slots=True)
class GraphicsShader:
    function: FunctionType
    stage: str

    @property
    def __name__(self):
        return self.function.__name__


def compute(*, workgroup_size=(1, 1, 1), capabilities=()):
    """Declare a Python function as a compute-shader entry point."""
    try:
        size = tuple(int(value) for value in workgroup_size)
    except (TypeError, ValueError) as error:
        raise ShaderTypeError("workgroup_size must contain three integers") from error
    if len(size) != 3 or any(value < 1 for value in size):
        raise ShaderTypeError("workgroup_size must contain three positive integers")
    declared_capabilities = tuple(capabilities)
    supported = {"subgroup_ballot", "shader_reorder"}
    unknown = set(declared_capabilities) - supported
    if unknown:
        raise ShaderTypeError(f"unsupported shader capabilities: {sorted(unknown)!r}")

    def decorate(function):
        if not isinstance(function, FunctionType):
            raise ShaderTypeError("@compute can only decorate a Python function")
        return ComputeShader(function, size, declared_capabilities)

    return decorate


def _graphics(stage):
    def decorate(function):
        if not isinstance(function, FunctionType):
            raise ShaderTypeError(f"@{stage} can only decorate a Python function")
        return GraphicsShader(function, stage)
    return decorate


vertex = _graphics("vertex")
fragment = _graphics("fragment")


def function(function):
    """Declare a reusable, typed shader helper function."""
    if not isinstance(function, FunctionType):
        raise ShaderTypeError("@function can only decorate a Python function")
    return ShaderFunction(function)


def external(function):
    if not isinstance(function, FunctionType):
        raise ShaderTypeError("@external can only decorate a Python function")
    return ExternalFunction(function)
