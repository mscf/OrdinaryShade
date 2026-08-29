"""Public compilation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

from ..targets import emit_glsl, emit_wgsl
from ..diagnostics import SourceMapEntry, annotate_error, module_source_map
from ..entrypoints import GraphicsShader
from ..errors import CompilerUnavailableError, ShaderCompilationError, ShaderError, ShaderTypeError
from ..lowering import lower, lower_external, lower_function, lower_graphics
from ..reflection import GraphicsPipelineReflection, ResourceReflection, ShaderReflection, StageIOReflection
from ..validation import validate_wgsl
from ..types import AccelerationStructure, ComparisonSampler, PushConstants, RuntimeArrayType, SampledDepthTexture2D, SampledTexture2D, SampledTexture2DArray, SampledTexture3DArray, Sampler, ShaderType, StorageBuffer, StorageImage, StorageRecord, StructType, UniformBuffer


@dataclass(frozen=True, slots=True)
class CompiledShader:
    target: str
    source: str
    binary: bytes | None
    reflection: ShaderReflection
    cache_key: str = ""
    source_map: tuple[SourceMapEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledFunction:
    source: str
    name: str
    target: str = "glsl"
    cache_key: str = ""
    source_map: tuple[SourceMapEntry, ...] = ()


def _cache_key(target: str, source: str) -> str:
    """Return a stable content identity suitable for persistent caches."""
    payload = b"ordinaryshade\0v1\0" + target.encode() + b"\0" + source.encode()
    return hashlib.sha256(payload).hexdigest()


def _compiled_shader(target, source, binary, reflection, shader):
    return CompiledShader(
        target, source, binary, reflection, _cache_key(target, source),
        module_source_map(source, shader),
    )


def link_graphics(vertex_shader, fragment_shader):
    """Validate and reflect a vertex/fragment interface pair."""
    vertex_reflection = vertex_shader.reflection
    fragment_reflection = fragment_shader.reflection
    if vertex_reflection.stage != "vertex" or fragment_reflection.stage != "fragment":
        raise ShaderTypeError("link_graphics() expects compiled vertex and fragment shaders")
    vertex_outputs = {
        item.location: item for item in vertex_reflection.outputs
        if item.location is not None
    }
    varyings = []
    for item in fragment_reflection.inputs:
        if item.location is None:
            continue
        producer = vertex_outputs.get(item.location)
        if producer is None:
            raise ShaderTypeError(f"fragment location {item.location} has no vertex output")
        if producer.type != item.type:
            raise ShaderTypeError(
                f"graphics location {item.location} type mismatch: "
                f"vertex {producer.type}, fragment {item.type}"
            )
        varyings.append(item)
    return GraphicsPipelineReflection(
        vertex_reflection, fragment_reflection, tuple(varyings),
    )


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
        if isinstance(resource.type, SampledDepthTexture2D):
            return ResourceReflection(
                resource.name, "sampled_depth_texture_2d", "depth",
                "read", resource.set, resource.binding,
            )
        if isinstance(resource.type, SampledTexture2D):
            return ResourceReflection(
                resource.name, "sampled_texture_2d", "rgba",
                "read", resource.set, resource.binding,
            )
        if isinstance(resource.type, Sampler):
            return ResourceReflection(
                resource.name, "sampler", "sampler",
                "read", resource.set, resource.binding,
            )
        if isinstance(resource.type, ComparisonSampler):
            return ResourceReflection(
                resource.name, "comparison_sampler", "sampler_comparison",
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


def _graphics_reflection(module):
    def reflected(item):
        return StageIOReflection(
            item.name, item.type_name, item.location, item.builtin,
        )
    resources = _reflection(type("ComputeReflectionView", (), {
        "resources": module.resources, "workgroup_size": (1, 1, 1),
    })()).resources
    return ShaderReflection(
        module.stage, "main", (1, 1, 1), resources,
        tuple(reflected(item) for item in module.inputs),
        tuple(reflected(item) for item in module.outputs),
    )


def _spirv(source, compiler, stage="compute"):
    executable = shutil.which(compiler)
    if executable is None:
        raise CompilerUnavailableError(
            f"could not locate {compiler!r}; install glslangValidator or compile to GLSL"
        )
    with tempfile.TemporaryDirectory(prefix="ordinaryshade-") as directory:
        suffix = {"compute": "comp", "vertex": "vert", "fragment": "frag"}[stage]
        source_path = Path(directory) / f"shader.{suffix}"
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
    try:
        module = (
            lower_graphics(shader) if isinstance(shader, GraphicsShader)
            else lower(shader, helpers=helpers, externals=externals)
        )
    except ShaderError as error:
        raise annotate_error(error, shader) from error.__cause__
    reflection = (
        _graphics_reflection(module)
        if isinstance(shader, GraphicsShader) else _reflection(module)
    )
    if target == "glsl":
        source = emit_glsl(module)
        return _compiled_shader(target, source, None, reflection, shader)
    if target == "wgsl":
        source = emit_wgsl(module)
        if validate:
            validate_wgsl(source, validator=wgsl_validator)
        return _compiled_shader(target, source, None, reflection, shader)
    if target == "spirv":
        source = emit_glsl(module)
        return _compiled_shader(
            target, source, _spirv(source, spirv_compiler, reflection.stage), reflection, shader,
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
        source = declarations + emit_glsl(module)
        return CompiledFunction(source, module.name, target, _cache_key(target, source), module_source_map(source, shader))
    if target == "wgsl":
        if external_modules or declared_values:
            raise ShaderTypeError("WGSL does not support external ABI declarations")
        source = emit_wgsl(module)
        if validate:
            validate_wgsl(source, validator=wgsl_validator)
        return CompiledFunction(source, module.name, target, _cache_key(target, source), module_source_map(source, shader))
    raise ShaderTypeError("function target must be 'glsl' or 'wgsl'")
