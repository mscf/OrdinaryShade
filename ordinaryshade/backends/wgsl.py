"""WebGPU Shading Language emitter for Ordinary Shade IR."""

from __future__ import annotations

from ..errors import ShaderTypeError
from ..ir import (
    Assign, Attribute, Binary, Break, Call, Compare, Conditional, Continue,
    ExpressionStatement, ForRange, FunctionModule, If, Let, Literal, Name,
    Return, Subscript, Unary, While,
)
from ..types import (
    AccelerationStructure, FixedArrayType, PushConstants, RuntimeArrayType, SampledTexture2DArray, SampledTexture3DArray, StorageBuffer, StorageImage, StorageImageArray, StorageRecord, StructType,
    UniformBuffer,
)


_TYPES = {
    "float": "f32",
    "int": "i32",
    "uint": "u32",
    "bool": "bool",
    "vec2": "vec2<f32>",
    "vec3": "vec3<f32>",
    "vec4": "vec4<f32>",
    "bvec2": "vec2<bool>",
    "bvec3": "vec3<bool>",
    "bvec4": "vec4<bool>",
    "ivec2": "vec2<i32>",
    "ivec3": "vec3<i32>",
    "ivec4": "vec4<i32>",
    "uvec2": "vec2<u32>",
    "uvec3": "vec3<u32>",
    "uvec4": "vec4<u32>",
    "mat3": "mat3x3<f32>",
    "mat4": "mat4x4<f32>",
    "f32": "f32",
    "i32": "i32",
    "u32": "u32",
    "boolean": "bool",
    "void": "void",
}

_STORAGE_FORMATS = {
    "rgba16f": "rgba16float",
    "rgba32f": "rgba32float",
    "rgba8": "rgba8unorm",
    "rgba8_snorm": "rgba8snorm",
    "r32f": "r32float",
    "r32i": "r32sint",
    "r32ui": "r32uint",
}

_INTRINSICS = {
    "mix": "mix",
    "minimum": "min",
    "maximum": "max",
    "clamp": "clamp",
    "dot": "dot",
    "normalize": "normalize",
    "power": "pow",
    "round": "round",
    "absolute": "abs",
    "sign": "sign",
    "sqrt": "sqrt",
    "exp": "exp",
    "logarithm": "log",
    "ceiling": "ceil",
    "length": "length",
    "cross": "cross",
    "refract": "refract",
    "cosine": "cos",
    "sine": "sin",
    "arctangent2": "atan2",
    "arccosine": "acos",
    "fraction": "fract",
    "any_value": "any", "subgroup_ballot": "subgroupBallot",
    "subgroup_ballot_bit_count": "subgroupBallotBitCount",
    "subgroup_ballot_exclusive_bit_count": "subgroupBallotExclusiveBitCount",
    "subgroup_elect": "subgroupElect",
    "subgroup_broadcast_first": "subgroupBroadcastFirst",
    "atomic_add": "atomicAdd",
    "pack_half2x16": "pack2x16float",
    "unpack_half2x16": "unpack2x16float",
    "pack_unorm2x16": "pack2x16unorm",
    "unpack_unorm2x16": "unpack2x16unorm",
    "exp2": "exp2", "floor": "floor", "log2": "log2",
    "memory_barrier_buffer": "storageBarrier",
    "workgroup_barrier": "workgroupBarrier",
    "pack_unorm4x8": "pack4x8unorm",
    "float_bits_to_uint": "bitcast<u32>",
    "uint_bits_to_float": "bitcast<f32>",
    "bitfield_reverse": "reverseBits",
    "vec2": "vec2<f32>",
    "vec3": "vec3<f32>",
    "vec4": "vec4<f32>",
    "bvec2": "vec2<bool>",
    "bvec3": "vec3<bool>",
    "bvec4": "vec4<bool>",
    "ivec2": "vec2<i32>",
    "ivec3": "vec3<i32>",
    "ivec4": "vec4<i32>",
    "uvec2": "vec2<u32>",
    "uvec3": "vec3<u32>",
    "uvec4": "vec4<u32>",
    "mat3": "mat3x3<f32>",
    "mat4": "mat4x4<f32>",
    "f32": "f32",
    "i32": "i32",
    "u32": "u32",
    "boolean": "bool",
}

# WGSL reserves both its current grammar and a larger set of words retained for
# future language evolution. Backend-local mangling keeps the Python API and
# reflection names stable while ensuring emitted modules remain legal WGSL.
_RESERVED = {
    "alias", "asm", "attribute", "auto", "await", "become", "binding_array",
    "break", "case", "catch", "class", "co_await", "co_return", "co_yield",
    "const", "const_assert", "const_cast", "consteval", "constexpr", "continue",
    "continuing", "crate", "debugger", "decltype", "default", "delete",
    "diagnostic", "discard", "do", "dynamic_cast", "else", "enable", "enum",
    "explicit", "export", "extends", "extern", "external", "false", "final",
    "finally", "fn", "for", "friend", "from", "fxgroup", "get", "goto",
    "groupshared", "handle", "highp", "if", "impl", "implements", "import",
    "inline", "instanceof", "interface", "layout", "let", "loop", "lowp",
    "macro", "match", "mediump", "meta", "mod", "module", "move", "mut",
    "mutable", "namespace", "new", "nil", "noinline", "nointerpolation",
    "noperspective", "null", "nullptr", "of", "operator", "override", "package",
    "packoffset", "partition", "pass", "patch", "pixelfragment", "precise",
    "precision", "premerge", "priv", "protected", "pub", "public", "readonly",
    "ref", "regardless", "register", "reinterpret_cast", "requires", "resource",
    "restrict", "return", "self", "set", "shared", "sizeof", "smooth", "snorm",
    "static", "static_assert", "static_cast", "std", "struct", "subroutine",
    "super", "switch", "target", "template", "this", "thread_local", "throw",
    "trait", "true", "try", "type", "typedef", "typeof", "union", "unless",
    "unorm", "unsafe", "unsized", "use", "using", "var", "varying", "virtual",
    "void", "volatile", "wgsl", "where", "while", "with", "writeonly", "yield",
}


def _identifier(name):
    return f"{name}_" if name in _RESERVED else name


def _type(type_name):
    try:
        return _TYPES[type_name]
    except KeyError as error:
        raise ShaderTypeError(f"WGSL backend cannot emit type {type_name!r}") from error


def _value_type(type_name):
    return type_name if type_name not in _TYPES else _type(type_name)


def _field_type(field):
    if isinstance(field.type, RuntimeArrayType):
        element = field.type.element_type
        name = element.name if isinstance(element, StructType) else _type(element.name)
        return f"array<{name}>"
    if isinstance(field.type, StructType):
        return field.type.name
    if isinstance(field.type, FixedArrayType):
        element = field.type.element_type
        name = element.name if isinstance(element, StructType) else _type(element.name)
        return f"array<{name}, {field.type.count}>"
    return _type(field.type.name)


def _is_ordinaryshade_attribute(value):
    return (
        isinstance(value, Attribute)
        and isinstance(value.value, Name)
        and value.value.value in {"osh", "ordinaryshade"}
    )


def _expression(value):
    if isinstance(value, Name):
        return _identifier(value.value)
    if isinstance(value, Literal):
        if isinstance(value.value, bool):
            return "true" if value.value else "false"
        if isinstance(value.value, float):
            text = repr(value.value)
            return text if "." in text or "e" in text.lower() else text + ".0"
        return str(value.value)
    if isinstance(value, Attribute):
        if (
            isinstance(value.value, Attribute)
            and isinstance(value.value.value, Name)
            and value.value.value.value in {"osh", "ordinaryshade"}
            and value.value.attribute in {
                "global_invocation_id", "local_invocation_id",
                "workgroup_id", "num_workgroups",
            }
        ):
            return f"{value.value.attribute}.{value.attribute}"
        if (
            isinstance(value.value, Name)
            and value.value.value in {"osh", "ordinaryshade"}
            and value.attribute == "local_invocation_index"
        ):
            return "local_invocation_index"
        if (
            isinstance(value.value, Attribute)
            and isinstance(value.value.value, Name)
            and value.value.value.value in {"osh", "ordinaryshade"}
            and value.value.attribute == "workgroup_size"
        ):
            raise ShaderTypeError(
                "WGSL workgroup-size introspection requires specialization"
            )
        if _is_ordinaryshade_attribute(value) and value.attribute in _INTRINSICS:
            return _INTRINSICS[value.attribute]
        return f"{_expression(value.value)}.{value.attribute}"
    if isinstance(value, Subscript):
        return f"{_expression(value.value)}[{_expression(value.index)}]"
    if isinstance(value, Binary):
        if value.operator == "**":
            return f"pow({_expression(value.left)}, {_expression(value.right)})"
        return f"({_expression(value.left)} {value.operator} {_expression(value.right)})"
    if isinstance(value, Unary):
        if value.operator == "+":
            return _expression(value.value)
        return f"({value.operator}{_expression(value.value)})"
    if isinstance(value, Conditional):
        return (
            f"select({_expression(value.when_false)}, "
            f"{_expression(value.when_true)}, {_expression(value.condition)})"
        )
    if isinstance(value, Compare):
        return (
            f"({_expression(value.left)} {value.operator} "
            f"{_expression(value.right)})"
        )
    if isinstance(value, Call):
        if (
            _is_ordinaryshade_attribute(value.function)
            and value.function.attribute == "select"
            and len(value.arguments) == 3
        ):
            condition, when_true, when_false = map(_expression, value.arguments)
            # WGSL select(false_value, true_value, condition) has a different
            # order from Ordinary Shade's select(condition, yes, no).
            return f"select({when_false}, {when_true}, {condition})"
        arguments = ", ".join(_expression(item) for item in value.arguments)
        if isinstance(value.function, Attribute):
            owner = _expression(value.function.value)
            if value.function.attribute == "load":
                return f"textureLoad({owner}, {arguments})"
            if value.function.attribute == "store":
                return f"textureStore({owner}, {arguments})"
            if value.function.attribute == "size":
                return f"vec2<i32>(textureDimensions({owner}))"
        return f"{_expression(value.function)}({arguments})"
    raise ShaderTypeError(f"WGSL backend cannot emit {type(value).__name__}")


def _statement(value, indent=1):
    prefix = "    " * indent
    if isinstance(value, Let):
        if value.type_name.startswith("shared:"):
            return []
        if value.type_name.startswith("local_array:"):
            _, element_type, count = value.type_name.split(":", 2)
            return [
                f"{prefix}var {_identifier(value.name)}: "
                f"array<{_value_type(element_type)}, {count}>;"
            ]
        expression = _expression(value.value)
        if expression == "global_invocation_id.xy":
            expression = f"vec2<i32>({expression})"
        return [
            f"{prefix}let {_identifier(value.name)}: "
            f"{_value_type(value.type_name)} = {expression};"
        ]
    if isinstance(value, ExpressionStatement):
        return [f"{prefix}{_expression(value.value)};"]
    if isinstance(value, Return):
        suffix = f" {_expression(value.value)}" if value.value is not None else ""
        return [f"{prefix}return{suffix};"]
    if isinstance(value, Continue):
        return [f"{prefix}continue;"]
    if isinstance(value, Break):
        return [f"{prefix}break;"]
    if isinstance(value, While):
        lines = [f"{prefix}while ({_expression(value.condition)}) {{"]
        for statement in value.body:
            lines.extend(_statement(statement, indent + 1))
        lines.append(f"{prefix}}}")
        return lines
    if isinstance(value, Assign):
        return [f"{prefix}{_expression(value.target)} = {_expression(value.value)};"]
    if isinstance(value, ForRange):
        condition = "<" if value.step > 0 else ">"
        variable = _identifier(value.variable)
        start = _expression(value.start)
        stop = _expression(value.stop)
        lines = [
            f"{prefix}for (var {variable}: i32 = {start}; "
            f"{variable} {condition} {stop}; {variable} += {value.step}) {{"
        ]
        for statement in value.body:
            lines.extend(_statement(statement, indent + 1))
        lines.append(f"{prefix}}}")
        return lines
    if isinstance(value, If):
        lines = [f"{prefix}if ({_expression(value.condition)}) {{"]
        for statement in value.body:
            lines.extend(_statement(statement, indent + 1))
        lines.append(f"{prefix}}}")
        if value.else_body:
            lines[-1] += " else {"
            for statement in value.else_body:
                lines.extend(_statement(statement, indent + 1))
            lines.append(f"{prefix}}}")
        return lines
    raise ShaderTypeError(f"WGSL backend cannot emit {type(value).__name__}")


def emit_wgsl(module):
    """Emit a lowered Ordinary Shade module as WGSL source."""
    if isinstance(module, FunctionModule):
        if any(parameter.qualifier != "in" for parameter in module.parameters):
            raise ShaderTypeError("WGSL does not support inout parameters")
        parameters = ", ".join(
            f"{_identifier(parameter.name)}: {_value_type(parameter.type_name)}"
            for parameter in module.parameters
        )
        result = (
            "" if module.return_type == "void"
            else f" -> {_value_type(module.return_type)}"
        )
        shared_values = [
            statement for statement in module.statements
            if isinstance(statement, Let)
            and statement.type_name.startswith("shared:")
        ]
        lines = [
            *(
                f"var<workgroup> {_identifier(value.name)}: "
                f"{_value_type(value.type_name.split(':', 1)[1])};"
                for value in shared_values
            ),
            f"fn {_identifier(module.name)}({parameters}){result} {{",
        ]
        for statement in module.statements:
            lines.extend(_statement(statement))
        lines.extend(("}", ""))
        return "\n".join(lines)

    lines = []
    if module.externals:
        raise ShaderTypeError("WGSL does not support external function declarations")
    if "subgroup_ballot" in module.capabilities:
        raise ShaderTypeError("WGSL subgroup ballot support is not yet available")
    if "shader_reorder" in module.capabilities:
        raise ShaderTypeError("WGSL shader invocation reordering is not available")
    structures = {}
    def register_module_structure(candidate):
        if not isinstance(candidate, StructType) or candidate.name in structures:
            return
        for field in candidate.fields:
            nested = (
                field.type.element_type
                if isinstance(field.type, (RuntimeArrayType, FixedArrayType)) else field.type
            )
            register_module_structure(nested)
        structures[candidate.name] = candidate
    for item in module.structures:
        register_module_structure(item)
    for resource in module.resources:
        if isinstance(resource.type, StorageImageArray):
            raise ShaderTypeError(
                "WGSL storage-image descriptor arrays are not yet available"
            )
        if isinstance(resource.type, SampledTexture3DArray):
            raise ShaderTypeError(
                "WGSL does not support Vulkan combined-sampler descriptor arrays"
            )
        if isinstance(resource.type, SampledTexture2DArray):
            raise ShaderTypeError(
                "WGSL does not support Vulkan combined-sampler descriptor arrays"
            )
        if isinstance(resource.type, AccelerationStructure):
            raise ShaderTypeError("WGSL does not support acceleration structures or ray queries")
        if isinstance(resource.type, PushConstants):
            raise ShaderTypeError(
                "WGSL has no push-constant resource; use a uniform buffer"
            )
        struct_type = getattr(
            resource.type, "element_type", getattr(resource.type, "struct_type", None)
        )
        def register_structure(candidate):
            if not isinstance(candidate, StructType) or candidate.name in structures:
                return
            for field in candidate.fields:
                nested = field.type.element_type if isinstance(field.type, (RuntimeArrayType, FixedArrayType)) else field.type
                register_structure(nested)
            structures[candidate.name] = candidate
        register_structure(struct_type)
    for struct_type in structures.values():
        lines.append(f"struct {struct_type.name} {{")
        lines.extend(
            f"    {_identifier(field.name)}: {_field_type(field)},"
            for field in struct_type.fields
        )
        lines.extend(("}", ""))
    for resource in module.resources:
        if isinstance(resource.type, StorageImage):
            try:
                format_name = _STORAGE_FORMATS[resource.type.format]
            except KeyError as error:
                raise ShaderTypeError(
                    f"WGSL backend cannot emit storage format {resource.type.format!r}"
                ) from error
            lines.append(
                f"@group({resource.set}) @binding({resource.binding}) "
                f"var {_identifier(resource.name)}: texture_storage_2d<"
                f"{format_name}, {resource.type.access}>;"
            )
        elif isinstance(resource.type, StorageBuffer):
            element_type = resource.type.element_type
            element_name = (
                element_type.name
                if isinstance(element_type, StructType)
                else _type(element_type.name)
            )
            access = (
                "read" if resource.type.access == "read" else "read_write"
            )
            lines.append(
                f"@group({resource.set}) @binding({resource.binding}) "
                f"var<storage, {access}> {_identifier(resource.name)}: "
                f"array<{element_name}>;"
            )
        elif isinstance(resource.type, StorageRecord):
            # WGSL has no write-only storage access. A write-only source
            # contract therefore lowers to its narrowest WGSL equivalent.
            access = "read" if resource.type.access == "read" else "read_write"
            lines.append(
                f"@group({resource.set}) @binding({resource.binding}) "
                f"var<storage, {access}> {_identifier(resource.name)}: "
                f"{resource.type.struct_type.name};"
            )
        elif isinstance(resource.type, UniformBuffer):
            lines.append(
                f"@group({resource.set}) @binding({resource.binding}) "
                f"var<uniform> {_identifier(resource.name)}: "
                f"{resource.type.struct_type.name};"
            )
    for helper in module.functions:
        lines.append("")
        lines.extend(emit_wgsl(helper).rstrip().splitlines())
    shared_values = [
        statement for statement in module.statements
        if isinstance(statement, Let) and statement.type_name.startswith("shared:")
    ]
    for value in shared_values:
        lines.append(
            f"var<workgroup> {_identifier(value.name)}: "
            f"{_value_type(value.type_name.split(':', 1)[1])};"
        )
    if lines:
        lines.append("")
    x, y, z = module.workgroup_size
    lines.extend((
        f"@compute @workgroup_size({x}, {y}, {z})",
        "fn main(",
        "    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,",
        "    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,",
        "    @builtin(local_invocation_index) local_invocation_index: u32,",
        "    @builtin(workgroup_id) workgroup_id: vec3<u32>,",
        "    @builtin(num_workgroups) num_workgroups: vec3<u32>,",
        ") {",
    ))
    for statement in module.statements:
        lines.extend(_statement(statement))
    lines.extend(("}", ""))
    return "\n".join(lines)
