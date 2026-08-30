"""Public shader and resource type declarations."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

from ..errors import ShaderTypeError


@dataclass(frozen=True, slots=True)
class ShaderType:
    """A statically known value type in the Ordinary Shade language."""

    name: str

    def __call__(self, *values):
        raise RuntimeError(f"{self.name} construction is only valid in shader source")


@dataclass(frozen=True, slots=True)
class QualifiedType:
    type: ShaderType | StructType
    qualifier: str

    @property
    def name(self):
        return self.type.name


@dataclass(frozen=True, slots=True)
class StageIOType:
    """A graphics-stage value attached to a location or pipeline builtin."""

    type: ShaderType | StructType
    location: int | None = None
    builtin: str | None = None
    invariant: bool = False

    @property
    def name(self):
        return self.type.name


def location(value_type, index: int):
    if not isinstance(value_type, (ShaderType, StructType)):
        raise ShaderTypeError("stage locations require a shader value type")
    if not isinstance(index, int) or index < 0:
        raise ShaderTypeError("stage locations require a non-negative integer")
    return StageIOType(value_type, location=index)


def builtin(value_type, name: str):
    supported = {"position", "vertex_index", "instance_index", "front_facing", "frag_depth"}
    if not isinstance(value_type, (ShaderType, StructType)):
        raise ShaderTypeError("stage builtins require a shader value type")
    if name not in supported:
        raise ShaderTypeError(f"unsupported graphics builtin {name!r}")
    return StageIOType(value_type, builtin=name)


def invariant(stage_io):
    """Require an output to produce invariant values across pipelines.

    This is primarily useful for vertex ``position`` outputs reused by depth
    and shading passes.  Backends emit the target language's invariant
    interface qualifier rather than relying on optimizer coincidence.
    """
    if not isinstance(stage_io, StageIOType):
        raise ShaderTypeError("invariant() requires a location() or builtin() type")
    return StageIOType(
        stage_io.type, stage_io.location, stage_io.builtin, invariant=True,
    )


def inout(value_type):
    """Declare a mutable shader-function parameter."""
    if not isinstance(value_type, (ShaderType, StructType)):
        raise ShaderTypeError("inout parameters require a shader value or structure type")
    return QualifiedType(value_type, "inout")


def opaque_type(name: str):
    """Reference a backend ABI type declared by the embedding shader."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ShaderTypeError("opaque shader type names must be identifiers")
    return ShaderType(name)


f32 = ShaderType("float")
i32 = ShaderType("int")
u32 = ShaderType("uint")
boolean = ShaderType("bool")
void = ShaderType("void")
vec2 = ShaderType("vec2")
vec3 = ShaderType("vec3")
vec4 = ShaderType("vec4")
bvec2 = ShaderType("bvec2")
bvec3 = ShaderType("bvec3")
bvec4 = ShaderType("bvec4")
ivec2 = ShaderType("ivec2")
ivec3 = ShaderType("ivec3")
ivec4 = ShaderType("ivec4")
uvec2 = ShaderType("uvec2")
uvec3 = ShaderType("uvec3")
uvec4 = ShaderType("uvec4")
mat3 = ShaderType("mat3")
mat4 = ShaderType("mat4")
ray_query = ShaderType("rayQueryEXT")

@dataclass(frozen=True, slots=True)
class StructField:
    name: str
    type: ShaderType | StructType | RuntimeArrayType | FixedArrayType | StageIOType


@dataclass(frozen=True, slots=True)
class FixedArrayType:
    element_type: ShaderType | StructType
    count: int

    @property
    def name(self):
        return f"fixed_array:{self.element_type.name}:{self.count}"


def array(element_type, count: int):
    if not isinstance(element_type, (ShaderType, StructType)):
        raise ShaderTypeError("arrays require a shader value or structure type")
    if not isinstance(count, int) or count <= 0:
        raise ShaderTypeError("array count must be a positive integer")
    return FixedArrayType(element_type, count)


@dataclass(frozen=True, slots=True)
class RuntimeArrayType:
    element_type: ShaderType | StructType

    @property
    def name(self):
        return f"runtime_array:{self.element_type.name}"


def runtime_array(element_type):
    if not isinstance(element_type, (ShaderType, StructType)):
        raise ShaderTypeError("runtime arrays require a shader value or structure type")
    return RuntimeArrayType(element_type)


def local_array(element_type, count: int):
    """Declare an uninitialized fixed-size array in function-local storage."""
    if not isinstance(element_type, (ShaderType, StructType)):
        raise ShaderTypeError("local arrays require a shader value or structure type")
    if not isinstance(count, int) or count <= 0:
        raise ShaderTypeError("local array count must be a positive integer")
    raise RuntimeError("local_array() is only valid in shader source")


def shared(element_type):
    """Declare a workgroup-shared value in a compute entry point."""
    if not isinstance(element_type, (ShaderType, StructType)):
        raise ShaderTypeError("shared values require a shader value or structure type")
    raise RuntimeError("shared() is only valid in shader source")


@dataclass(frozen=True, slots=True)
class StructType:
    name: str
    fields: tuple[StructField, ...]

    def __call__(self, *values):
        raise RuntimeError(f"{self.name} construction is only valid in shader source")


def structure(cls):
    """Declare a fixed-layout shader structure from annotated fields."""
    if not inspect.isclass(cls):
        raise ShaderTypeError("@structure can only decorate a class")
    annotations = inspect.get_annotations(cls, eval_str=True)
    if not annotations:
        raise ShaderTypeError("shader structures require at least one annotated field")
    fields = []
    for index, (name, field_type) in enumerate(annotations.items()):
        if not isinstance(field_type, (ShaderType, StructType, RuntimeArrayType, FixedArrayType, StageIOType)):
            raise ShaderTypeError(
                f"structure field {name!r} requires a shader value type"
            )
        if isinstance(field_type, RuntimeArrayType) and index != len(annotations) - 1:
            raise ShaderTypeError("a runtime array must be the final structure field")
        fields.append(StructField(name, field_type))
    return StructType(cls.__name__, tuple(fields))


def mix(*values):
    raise RuntimeError("mix() is only valid in shader source")


def minimum(*values):
    raise RuntimeError("minimum() is only valid in shader source")


def maximum(*values):
    raise RuntimeError("maximum() is only valid in shader source")


def absolute(*values):
    raise RuntimeError("absolute() is only valid in shader source")


def sign(*values):
    raise RuntimeError("sign() is only valid in shader source")


def sqrt(*values):
    raise RuntimeError("sqrt() is only valid in shader source")


def exp(*values):
    raise RuntimeError("exp() is only valid in shader source")


def logarithm(*values):
    raise RuntimeError("logarithm() is only valid in shader source")


def ceiling(*values):
    raise RuntimeError("ceiling() is only valid in shader source")


def length(*values):
    raise RuntimeError("length() is only valid in shader source")


def cross(*values):
    raise RuntimeError("cross() is only valid in shader source")


def refract(*values):
    raise RuntimeError("refract() is only valid in shader source")


def cosine(*values):
    raise RuntimeError("cosine() is only valid in shader source")


def sine(*values):
    raise RuntimeError("sine() is only valid in shader source")


def arctangent2(*values):
    raise RuntimeError("arctangent2() is only valid in shader source")


def arccosine(*values):
    raise RuntimeError("arccosine() is only valid in shader source")


def fraction(*values):
    raise RuntimeError("fraction() is only valid in shader source")


def any_value(*values):
    raise RuntimeError("any_value() is only valid in shader source")


def subgroup_ballot(*values):
    raise RuntimeError("subgroup_ballot() is only valid in shader source")


def subgroup_ballot_bit_count(*values):
    raise RuntimeError("subgroup_ballot_bit_count() is only valid in shader source")


def subgroup_ballot_exclusive_bit_count(*values):
    raise RuntimeError(
        "subgroup_ballot_exclusive_bit_count() is only valid in shader source"
    )


def subgroup_elect(*values):
    raise RuntimeError("subgroup_elect() is only valid in shader source")


def subgroup_broadcast_first(*values):
    raise RuntimeError("subgroup_broadcast_first() is only valid in shader source")


def atomic_add(*values):
    raise RuntimeError("atomic_add() is only valid in shader source")


def workgroup_barrier(*values):
    raise RuntimeError("workgroup_barrier() is only valid in shader source")


def reorder_thread(*values):
    raise RuntimeError("reorder_thread() is only valid in shader source")


def exp2(*values):
    raise RuntimeError("exp2() is only valid in shader source")


def floor(*values):
    raise RuntimeError("floor() is only valid in shader source")


def log2(*values):
    raise RuntimeError("log2() is only valid in shader source")


def pack_half2x16(*values):
    raise RuntimeError("pack_half2x16() is only valid in shader source")


def unpack_half2x16(*values):
    raise RuntimeError("unpack_half2x16() is only valid in shader source")


def pack_unorm2x16(*values):
    raise RuntimeError("pack_unorm2x16() is only valid in shader source")


def unpack_unorm2x16(*values):
    raise RuntimeError("unpack_unorm2x16() is only valid in shader source")


def memory_barrier_buffer(*values):
    raise RuntimeError("memory_barrier_buffer() is only valid in shader source")


def clamp(*values):
    raise RuntimeError("clamp() is only valid in shader source")


def power(*values):
    raise RuntimeError("power() is only valid in shader source")


def round(*values):
    raise RuntimeError("round() is only valid in shader source")


def select(*values):
    raise RuntimeError("select() is only valid in shader source")


def pack_unorm4x8(*values):
    raise RuntimeError("pack_unorm4x8() is only valid in shader source")


def float_bits_to_uint(*values):
    raise RuntimeError("float_bits_to_uint() is only valid in shader source")


def uint_bits_to_float(*values):
    raise RuntimeError("uint_bits_to_float() is only valid in shader source")


def bitfield_reverse(*values):
    raise RuntimeError("bitfield_reverse() is only valid in shader source")


_IMAGE_FORMAT_TYPES = {
    "unformatted": "unformatted",
    "rgba16f": "rgba16f",
    "rgba32f": "rgba32f",
    "rgba8": "rgba8",
    "rgba8_snorm": "rgba8_snorm",
    "r32f": "r32f",
    "r32i": "r32i",
    "r32ui": "r32ui",
    "r11f_g11f_b10f": "r11f_g11f_b10f",
}


@dataclass(frozen=True, slots=True)
class StorageImage:
    """A two-dimensional GLSL storage-image binding declaration."""

    format: str
    access: str = "read_write"
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if self.format not in _IMAGE_FORMAT_TYPES:
            raise ShaderTypeError(f"unsupported storage image format: {self.format!r}")
        if self.access not in {"read", "write", "read_write"}:
            raise ShaderTypeError("storage image access must be read, write, or read_write")
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


def storage_image(format: str, *, access="read_write", set=0, binding=None):
    return StorageImage(format, access, set, binding)


@dataclass(frozen=True, slots=True)
class StorageImageArray(StorageImage):
    """A fixed-size Vulkan descriptor array of two-dimensional images."""

    count: int = 1

    def __post_init__(self):
        super(StorageImageArray, self).__post_init__()
        if not isinstance(self.count, int) or self.count < 1:
            raise ShaderTypeError("storage image array count must be positive")


def storage_image_array(
    format: str, count: int, *, access="read_write", set=0, binding=None,
):
    return StorageImageArray(format, access, set, binding, count)


@dataclass(frozen=True, slots=True)
class SampledTexture3DArray:
    """A fixed-size Vulkan array of combined 3D texture/sampler bindings."""

    count: int = 1
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if not isinstance(self.count, int) or self.count < 1:
            raise ShaderTypeError("sampled 3D texture array count must be positive")
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


def sampled_texture_3d_array(count: int, *, set=0, binding=None):
    return SampledTexture3DArray(count, set, binding)


@dataclass(frozen=True, slots=True)
class SampledTexture2DArray:
    """A fixed-size Vulkan array of combined 2D texture/sampler bindings."""

    count: int = 1
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if not isinstance(self.count, int) or self.count < 1:
            raise ShaderTypeError("sampled 2D texture array count must be positive")
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


def sampled_texture_2d_array(count: int, *, set=0, binding=None):
    return SampledTexture2DArray(count, set, binding)


@dataclass(frozen=True, slots=True)
class SampledTexture2D:
    """A portable separately bound two-dimensional sampled texture."""
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


def sampled_texture_2d(*, set=0, binding=None):
    return SampledTexture2D(set, binding)


@dataclass(frozen=True, slots=True)
class SampledDepthTexture2D:
    """A portable separately bound two-dimensional depth texture."""
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


def sampled_depth_texture_2d(*, set=0, binding=None):
    return SampledDepthTexture2D(set, binding)


@dataclass(frozen=True, slots=True)
class Sampler:
    """A portable filtering sampler resource."""
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


def sampler(*, set=0, binding=None):
    return Sampler(set, binding)


@dataclass(frozen=True, slots=True)
class ComparisonSampler:
    """A portable depth-comparison sampler resource."""
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


def comparison_sampler(*, set=0, binding=None):
    return ComparisonSampler(set, binding)


@dataclass(frozen=True, slots=True)
class AccelerationStructure:
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


def acceleration_structure(*, set=0, binding=None):
    return AccelerationStructure(set, binding)


@dataclass(frozen=True, slots=True)
class StorageBuffer:
    element_type: StructType | ShaderType
    access: str = "read_write"
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if not isinstance(self.element_type, (StructType, ShaderType)):
            raise ShaderTypeError(
                "storage buffer elements require a shader value or structure type"
            )
        if self.access not in {"read", "write", "read_write"}:
            raise ShaderTypeError(
                "storage buffer access must be read, write, or read_write"
            )
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


@dataclass(frozen=True, slots=True)
class StorageRecord:
    """A fixed-layout structure stored in a descriptor-backed storage block."""

    struct_type: StructType
    access: str = "read_write"
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if not isinstance(self.struct_type, StructType):
            raise ShaderTypeError("storage records require a shader structure")
        if self.access not in {"read", "write", "read_write"}:
            raise ShaderTypeError(
                "storage record access must be read, write, or read_write"
            )
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


@dataclass(frozen=True, slots=True)
class UniformBuffer:
    struct_type: StructType
    set: int = 0
    binding: int | None = None

    def __post_init__(self):
        if not isinstance(self.struct_type, StructType):
            raise ShaderTypeError("uniform buffers require a shader structure")
        if self.set < 0 or (self.binding is not None and self.binding < 0):
            raise ShaderTypeError("descriptor set and binding must be non-negative")


@dataclass(frozen=True, slots=True)
class PushConstants:
    struct_type: StructType

    def __post_init__(self):
        if not isinstance(self.struct_type, StructType):
            raise ShaderTypeError("push constants require a shader structure")


def storage_buffer(element_type, *, access="read_write", set=0, binding=None):
    return StorageBuffer(element_type, access, set, binding)


def storage_record(struct_type, *, access="read_write", set=0, binding=None):
    return StorageRecord(struct_type, access, set, binding)


def uniform_buffer(struct_type, *, set=0, binding=None):
    return UniformBuffer(struct_type, set, binding)


def push_constants(struct_type):
    return PushConstants(struct_type)


class _Builtin:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, component):
        raise RuntimeError(
            f"{self.name}.{component} is only valid inside an Ordinary Shade function"
        )


global_invocation_id = _Builtin("global_invocation_id")
local_invocation_id = _Builtin("local_invocation_id")
local_invocation_index = _Builtin("local_invocation_index")
workgroup_id = _Builtin("workgroup_id")
num_workgroups = _Builtin("num_workgroups")
workgroup_size = _Builtin("workgroup_size")
