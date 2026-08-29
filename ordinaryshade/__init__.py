"""Ordinary Shade public API."""

from .compiler import CompiledFunction, CompiledShader, compile, compile_function
from .entrypoints import ComputeShader, ExternalFunction, ShaderFunction, compute, external, function
from .errors import (
    CompilerUnavailableError, ShaderCompilationError, ShaderError,
    ShaderSyntaxError, ShaderTypeError,
)
from .types import (
    AccelerationStructure, FixedArrayType, PushConstants, QualifiedType, ShaderType, StorageBuffer, StorageImage, StorageImageArray, SampledTexture2DArray, SampledTexture3DArray, StorageRecord, StructField,
    StructType, RuntimeArrayType, UniformBuffer, acceleration_structure, absolute, any_value, atomic_add, boolean, bvec2, bvec3, bvec4, clamp, cosine, cross, exp, exp2, f32, floor, log2, memory_barrier_buffer,
    global_invocation_id, local_invocation_id, local_invocation_index, workgroup_id, num_workgroups, workgroup_size, i32, ivec2, ivec3, ivec4, mat3, mat4, maximum, minimum, mix, power, refract, logarithm, ceiling,
    pack_unorm4x8, float_bits_to_uint, uint_bits_to_float, bitfield_reverse, push_constants, round, select, sign, sqrt, storage_buffer, storage_image,
    storage_record, storage_image_array, sampled_texture_2d_array, sampled_texture_3d_array, structure, runtime_array, local_array, shared, sine, arctangent2, arccosine, fraction, length, pack_half2x16,
    array, inout, opaque_type, pack_unorm2x16, ray_query, subgroup_ballot,
    subgroup_ballot_bit_count, subgroup_ballot_exclusive_bit_count,
    subgroup_broadcast_first, subgroup_elect, reorder_thread, workgroup_barrier, u32, unpack_half2x16,
    unpack_unorm2x16, uniform_buffer, uvec2, uvec3, uvec4, vec2, vec3, vec4, void,
)
from .validation import validate_wgsl

__version__ = "0.1.0a0"

__all__ = [
    "AccelerationStructure", "CompiledFunction", "CompiledShader", "CompilerUnavailableError", "ComputeShader", "ExternalFunction", "FixedArrayType",
    "ShaderCompilationError", "ShaderError", "ShaderSyntaxError",
    "PushConstants", "QualifiedType", "ShaderFunction", "ShaderType", "ShaderTypeError",
    "StorageBuffer", "StorageImage", "StorageImageArray", "SampledTexture2DArray", "SampledTexture3DArray", "StorageRecord", "StructField", "StructType", "RuntimeArrayType", "UniformBuffer", "acceleration_structure", "absolute", "any_value", "atomic_add", "boolean",
    "array", "bvec2", "bvec3", "bvec4", "clamp", "cosine", "cross", "compile", "compile_function",
    "compute", "exp", "exp2", "external", "logarithm", "ceiling", "f32", "floor", "function", "inout", "length", "log2", "opaque_type",
    "global_invocation_id", "local_invocation_id", "local_invocation_index", "workgroup_id", "num_workgroups", "workgroup_size", "i32", "ivec2", "ivec3", "ivec4", "mat3", "mat4", "maximum",
    "memory_barrier_buffer", "minimum", "mix", "pack_half2x16", "pack_unorm2x16", "pack_unorm4x8", "float_bits_to_uint", "uint_bits_to_float", "bitfield_reverse", "power", "push_constants", "ray_query", "refract", "round", "select", "sign", "sqrt",
    "storage_buffer", "storage_record", "runtime_array", "local_array", "shared", "sampled_texture_2d_array", "sampled_texture_3d_array", "sine", "arctangent2", "arccosine", "fraction", "subgroup_ballot",
    "subgroup_ballot_bit_count", "subgroup_ballot_exclusive_bit_count",
    "subgroup_broadcast_first", "subgroup_elect", "reorder_thread", "workgroup_barrier",
    "storage_image", "storage_image_array", "structure", "u32", "uniform_buffer", "uvec2", "uvec3", "uvec4", "vec2",
    "unpack_half2x16", "unpack_unorm2x16", "validate_wgsl", "vec3", "vec4", "void",
]
