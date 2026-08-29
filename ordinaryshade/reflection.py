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
class ShaderReflection:
    stage: str
    entry_point: str
    workgroup_size: tuple[int, int, int]
    resources: tuple[ResourceReflection, ...]

