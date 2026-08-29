"""Public compilation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

from .backends import emit_glsl, emit_wgsl
from .errors import CompilerUnavailableError, ShaderCompilationError, ShaderTypeError
from .lowering import lower, lower_external, lower_function
from .reflection import ResourceReflection, ShaderReflection
from .validation import validate_wgsl
from .types import AccelerationStructure, PushConstants, RuntimeArrayType, SampledTexture2DArray, SampledTexture3DArray, ShaderType, StorageBuffer, StorageImage, StorageRecord, StructType, UniformBuffer


@dataclass(frozen=True, slots=True)
class CompiledShader:
    target: str
    source: str
    binary: bytes | None
    reflection: ShaderReflection


@dataclass(frozen=True, slots=True)
class CompiledFunction:
    source: str
    name: str
    target: str = "glsl"


def _reflection(module):
    def reflected(resource):
        if isinstance(resource.type, AccelerationStructure):
            return ResourceReflection(
                resource.name, "acceleration_structure", "acceleration_structure",
                "read", resource.set, resource.binding,
            )
        if isinstance(resource.type, StorageImage):
            return ResourceReflection(
                resource.name, "storage_image", resource.type.format,
                resource.type.access, resource.set, resource.binding,
            )
        if isinstance(resource.type, SampledTexture3DArray):
            return ResourceReflection(
                resource.name, "sampled_texture_3d_array", "r32f",
                "read", resource.set, resource.binding,
            )
        if isinstance(resource.type, SampledTexture2DArray):
            return ResourceReflection(
                resource.name, "sampled_texture_2d_array", "rgba",
                "read", resource.set, resource.binding,
            )
        if isinstance(resource.type, StorageBuffer):
            return ResourceReflection(
                resource.name, "storage_buffer", resource.type.element_type.name,
                resource.type.access, resource.set, resource.binding,
            )
        if isinstance(resource.type, StorageRecord):
            return ResourceReflection(
                resource.name, "storage_buffer", resource.type.struct_type.name,
                resource.type.access, resource.set, resource.binding,
            )
        if isinstance(resource.type, UniformBuffer):
            return ResourceReflection(
                resource.name, "uniform_buffer", resource.type.struct_type.name,
                "read", resource.set, resource.binding,
            )
        if isinstance(resource.type, PushConstants):
            return ResourceReflection(
                resource.name, "push_constants", resource.type.struct_type.name,
                "read", 0, -1,
            )
        raise ShaderTypeError(f"unsupported resource {type(resource.type).__name__}")
    return ShaderReflection(
        "compute", "main", module.workgroup_size,
        tuple(reflected(resource) for resource in module.resources),
    )


def _spirv(source, compiler):
    executable = shutil.which(compiler)
    if executable is None:
        raise CompilerUnavailableError(
            f"could not locate {compiler!r}; install glslangValidator or compile to GLSL"
        )
    with tempfile.TemporaryDirectory(prefix="ordinaryshade-") as directory:
        source_path = Path(directory) / "shader.comp"
        output_path = Path(directory) / "shader.spv"
        source_path.write_text(source)
        result = subprocess.run(
            (
                executable, "-V", "--target-env", "vulkan1.2",
                str(source_path), "-o", str(output_path),
            ),
            capture_output=True, text=True,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ShaderCompilationError(detail)
        return output_path.read_bytes()


def compile(
    shader,
    *,
    target="glsl",
    spirv_compiler="glslangValidator",
    validate=False,
    wgsl_validator="naga",
    helpers=(), externals=(),
):
    """Compile a decorated shader to GLSL, WGSL, or Vulkan SPIR-V."""
    module = lower(shader, helpers=helpers, externals=externals)
    reflection = _reflection(module)
    if target == "glsl":
        source = emit_glsl(module)
        return CompiledShader(target, source, None, reflection)
    if target == "wgsl":
        source = emit_wgsl(module)
        if validate:
            validate_wgsl(source, validator=wgsl_validator)
        return CompiledShader(target, source, None, reflection)
    if target == "spirv":
        source = emit_glsl(module)
        return CompiledShader(
            target, source, _spirv(source, spirv_compiler), reflection,
        )
    raise ShaderTypeError("target must be 'glsl', 'wgsl', or 'spirv'")


def compile_function(
    shader,
    *,
    target="glsl",
    validate=False,
    wgsl_validator="naga", externals=(), external_values=None,
    capabilities=(),
):
    """Compile a reusable typed helper function to backend source."""
    external_modules = tuple(lower_external(item) for item in externals)
    external_types = {item.name: item.return_type for item in external_modules}
    declared_values = {}
    declared_structures = {}
    def register_external_structure(candidate):
        if not isinstance(candidate, StructType) or candidate.name in declared_structures:
            return
        declared_structures[candidate.name] = candidate
        for field in candidate.fields:
            nested = getattr(field.type, "element_type", field.type)
            register_external_structure(nested)
    for name, value_type in dict(external_values or {}).items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ShaderTypeError("external value names must be identifiers")
        if not isinstance(value_type, (ShaderType, StructType, RuntimeArrayType)):
            raise ShaderTypeError("external values require shader value types")
        declared_values[name] = value_type.name
        register_external_structure(
            value_type.element_type
            if isinstance(value_type, RuntimeArrayType) else value_type
        )
    module = lower_function(
        shader, functions=external_types, external_types=declared_values,
        structures=declared_structures,
        capabilities=capabilities,
    )
    if target == "glsl":
        declarations = "".join(emit_glsl(item, declaration=True) for item in external_modules)
        return CompiledFunction(declarations + emit_glsl(module), module.name, target)
    if target == "wgsl":
        if external_modules or declared_values:
            raise ShaderTypeError("WGSL does not support external ABI declarations")
        source = emit_wgsl(module)
        if validate:
            validate_wgsl(source, validator=wgsl_validator)
        return CompiledFunction(source, module.name, target)
    raise ShaderTypeError("function target must be 'glsl' or 'wgsl'")
