"""Portable reflection records returned with compiled shaders."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceReflection:
    name: str
    kind: str
    format: str
    access: str
    set: int
    binding: int


@dataclass(frozen=True, slots=True)
class StageIOReflection:
    name: str
    type: str
    location: int | None = None
    builtin: str | None = None
    invariant: bool = False


@dataclass(frozen=True, slots=True)
class ShaderReflection:
    stage: str
    entry_point: str
    workgroup_size: tuple[int, int, int]
    resources: tuple[ResourceReflection, ...]
    inputs: tuple[StageIOReflection, ...] = ()
    outputs: tuple[StageIOReflection, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphicsPipelineReflection:
    vertex: ShaderReflection
    fragment: ShaderReflection
    varyings: tuple[StageIOReflection, ...]
