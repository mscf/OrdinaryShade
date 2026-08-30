"""Vulkan GLSL emitter for Ordinary Shade IR."""

from __future__ import annotations

from ..errors import ShaderTypeError
from ..ir import (
    Assign, Attribute, Binary, Break, Call, Compare, Conditional, Continue,
    ExpressionStatement, ForRange, FunctionModule, If, Let, Literal, Name,
    Return, Subscript, Unary, While, GraphicsModule,
)
from ..types import (
    AccelerationStructure, ComparisonSampler, FixedArrayType, PushConstants, RuntimeArrayType, SampledDepthTexture2D, SampledTexture2D, SampledTexture2DArray, SampledTexture3DArray, Sampler, StorageBuffer, StorageImage, StorageImageArray, StorageRecord, StructType,
    UniformBuffer,
)


_RESERVED = {
    "attribute", "buffer", "coherent", "common", "flat", "in", "input",
    "layout", "out", "output", "patch", "readonly", "resource", "shared",
    "sample", "smooth", "uniform", "varying", "volatile", "writeonly",
}


def _identifier(name):
    return f"{name}_" if name in _RESERVED else name


def _field_declaration(field):
    if isinstance(field.type, RuntimeArrayType):
        return f"{field.type.element_type.name} {_identifier(field.name)}[];"
    if isinstance(field.type, FixedArrayType):
        return (
            f"{field.type.element_type.name} {_identifier(field.name)}"
            f"[{field.type.count}];"
        )
    return f"{field.type.name} {_identifier(field.name)};"


def _expression(value):
    if isinstance(value, Name):
        return {
            "global_invocation_id": "gl_GlobalInvocationID",
            "local_invocation_id": "gl_LocalInvocationID",
            "local_invocation_index": "gl_LocalInvocationIndex",
            "workgroup_id": "gl_WorkGroupID",
            "num_workgroups": "gl_NumWorkGroups",
            "workgroup_size": "gl_WorkGroupSize",
        }.get(value.value, _identifier(value.value))
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
                "workgroup_id", "num_workgroups", "workgroup_size",
            }
        ):
            builtin = {
                "global_invocation_id": "gl_GlobalInvocationID",
                "local_invocation_id": "gl_LocalInvocationID",
                "workgroup_id": "gl_WorkGroupID",
                "num_workgroups": "gl_NumWorkGroups",
                "workgroup_size": "gl_WorkGroupSize",
            }[value.value.attribute]
            return f"{builtin}.{value.attribute}"
        if (
            isinstance(value.value, Name)
            and value.value.value in {"osh", "ordinaryshade"}
            and value.attribute == "local_invocation_index"
        ):
            return "gl_LocalInvocationIndex"
        if (
            isinstance(value.value, Name)
            and value.value.value in {"osh", "ordinaryshade"}
            and value.attribute in {
                "mix", "minimum", "maximum", "select", "clamp", "dot",
                "normalize", "power", "round", "absolute", "sign", "sqrt", "logarithm", "ceiling",
                "exp", "length", "cross", "refract", "cosine", "sine", "arctangent2", "arccosine", "fraction", "vec2", "vec3", "vec4", "bvec2",
                "any_value", "subgroup_ballot", "subgroup_ballot_bit_count",
                "subgroup_ballot_exclusive_bit_count", "subgroup_elect",
                "subgroup_broadcast_first", "atomic_add",
                "bvec3", "bvec4", "ivec2", "ivec3", "ivec4", "uvec2",
                "uvec3", "uvec4", "mat3", "mat4",
                "f32", "i32", "u32", "boolean", "pack_unorm4x8",
                "float_bits_to_uint", "uint_bits_to_float", "bitfield_reverse",
                "pack_half2x16", "unpack_half2x16", "pack_unorm2x16",
                "unpack_unorm2x16", "exp2", "floor", "log2",
                "memory_barrier_buffer",
                "workgroup_barrier", "reorder_thread",
            }
        ):
            return {
                "minimum": "min", "maximum": "max", "power": "pow", "logarithm": "log", "ceiling": "ceil",
                "pack_unorm4x8": "packUnorm4x8", "absolute": "abs",
                "float_bits_to_uint": "floatBitsToUint",
                "uint_bits_to_float": "uintBitsToFloat",
                "bitfield_reverse": "bitfieldReverse",
                "cosine": "cos", "sine": "sin", "arctangent2": "atan",
                "arccosine": "acos", "fraction": "fract",
                "any_value": "any", "subgroup_ballot": "subgroupBallot",
                "subgroup_ballot_bit_count": "subgroupBallotBitCount",
                "subgroup_ballot_exclusive_bit_count": "subgroupBallotExclusiveBitCount",
                "subgroup_elect": "subgroupElect",
                "subgroup_broadcast_first": "subgroupBroadcastFirst",
                "atomic_add": "atomicAdd",
                "pack_half2x16": "packHalf2x16",
                "unpack_half2x16": "unpackHalf2x16",
                "pack_unorm2x16": "packUnorm2x16",
                "unpack_unorm2x16": "unpackUnorm2x16",
                "memory_barrier_buffer": "memoryBarrierBuffer",
                "workgroup_barrier": "barrier",
                "reorder_thread": "reorderThreadNV",
                "f32": "float", "i32": "int", "u32": "uint", "boolean": "bool",
            }.get(
                value.attribute, value.attribute,
            )
        return f"{_expression(value.value)}.{_identifier(value.attribute)}"
    if isinstance(value, Subscript):
        return f"{_expression(value.value)}[{_expression(value.index)}]"
    if isinstance(value, Binary):
        if value.operator == "**":
            return f"pow({_expression(value.left)}, {_expression(value.right)})"
        return f"({_expression(value.left)} {value.operator} {_expression(value.right)})"
    if isinstance(value, Unary):
        return f"({value.operator}{_expression(value.value)})"
    if isinstance(value, Conditional):
        return (
            f"({_expression(value.condition)} ? {_expression(value.when_true)} "
            f": {_expression(value.when_false)})"
        )
    if isinstance(value, Compare):
        left = _expression(value.left)
        right = _expression(value.right)
        if value.vector:
            function = {
                "==": "equal", "!=": "notEqual", "<": "lessThan",
                "<=": "lessThanEqual", ">": "greaterThan",
                ">=": "greaterThanEqual",
            }[value.operator]
            return f"{function}({left}, {right})"
        return f"({left} {value.operator} {right})"
    if isinstance(value, Call):
        if (
            isinstance(value.function, Attribute)
            and isinstance(value.function.value, Name)
            and value.function.value.value in {"osh", "ordinaryshade"}
            and value.function.attribute == "select"
            and len(value.arguments) == 3
        ):
            condition, when_true, when_false = map(_expression, value.arguments)
            if value.selector_vector:
                return f"mix({when_false}, {when_true}, {condition})"
            return f"({condition} ? {when_true} : {when_false})"
        arguments = ", ".join(_expression(item) for item in value.arguments)
        if isinstance(value.function, Attribute):
            owner = _expression(value.function.value)
            ray_methods = {
                "initialize": "rayQueryInitializeEXT",
                "proceed": "rayQueryProceedEXT",
                "intersection_type": "rayQueryGetIntersectionTypeEXT",
                "intersection_t": "rayQueryGetIntersectionTEXT",
                "primitive_index": "rayQueryGetIntersectionPrimitiveIndexEXT",
                "instance_custom_index": "rayQueryGetIntersectionInstanceCustomIndexEXT",
                "barycentrics": "rayQueryGetIntersectionBarycentricsEXT",
            }
            if value.function.attribute in ray_methods:
                suffix = f", {arguments}" if arguments else ""
                return f"{ray_methods[value.function.attribute]}({owner}{suffix})"
            if value.function.attribute == "load":
                return f"imageLoad({owner}, {arguments})"
            if value.function.attribute == "sample" and len(value.arguments) == 2:
                index, coordinate = map(_expression, value.arguments)
                return f"texture({owner}[{index}], {coordinate}).r"
            if value.function.attribute == "sample_with" and len(value.arguments) == 2:
                sample, coordinate = map(_expression, value.arguments)
                return f"texture(sampler2D({owner}, {sample}), {coordinate})"
            if value.function.attribute == "sample_depth_with" and len(value.arguments) == 2:
                sample, coordinate = map(_expression, value.arguments)
                return f"texture(sampler2D({owner}, {sample}), {coordinate}).r"
            if value.function.attribute == "sample_compare_with" and len(value.arguments) == 3:
                sample, coordinate, reference = map(_expression, value.arguments)
                return f"texture(sampler2DShadow({owner}, {sample}), vec3({coordinate}, {reference}))"
            if value.function.attribute == "sample_lod" and len(value.arguments) == 3:
                index, coordinate, lod = map(_expression, value.arguments)
                return f"textureLod({owner}[nonuniformEXT({index})], {coordinate}, {lod})"
            if value.function.attribute == "size" and len(value.arguments) == 2:
                index, lod = map(_expression, value.arguments)
                return f"textureSize({owner}[nonuniformEXT({index})], {lod})"
            if value.function.attribute == "levels" and len(value.arguments) == 1:
                index = _expression(value.arguments[0])
                return f"textureQueryLevels({owner}[nonuniformEXT({index})])"
            if value.function.attribute == "store":
                return f"imageStore({owner}, {arguments})"
            if value.function.attribute == "size":
                return f"imageSize({owner})"
        return f"{_expression(value.function)}({arguments})"
    raise ShaderTypeError(f"GLSL backend cannot emit {type(value).__name__}")


def _statement(value, indent=1):
    prefix = "    " * indent
    if isinstance(value, Let):
        if value.type_name.startswith("shared:"):
            return []
        if value.type_name.startswith("local_array:"):
            _, element_type, count = value.type_name.split(":", 2)
            return [
                f"{prefix}{element_type} {_identifier(value.name)}[{count}];"
            ]
        if value.type_name == "rayQueryEXT":
            return [f"{prefix}rayQueryEXT {_identifier(value.name)};"]
        expression = _expression(value.value)
        if value.type_name == "ivec2" and expression == "gl_GlobalInvocationID.xy":
            expression = f"ivec2({expression})"
        return [
            f"{prefix}{value.type_name} {_identifier(value.name)} = {expression};"
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
        lines = [f"{prefix}while ({_expression(value.condition)})", f"{prefix}{{"]
        for statement in value.body:
            lines.extend(_statement(statement, indent + 1))
        lines.append(f"{prefix}}}")
        return lines
    if isinstance(value, Assign):
        return [f"{prefix}{_expression(value.target)} = {_expression(value.value)};"]
    if isinstance(value, ForRange):
        condition = "<" if value.step > 0 else ">"
        start = _expression(value.start)
        stop = _expression(value.stop)
        lines = [
            f"{prefix}for (int {_identifier(value.variable)} = {start}; "
            f"{_identifier(value.variable)} {condition} {stop}; "
            f"{_identifier(value.variable)} += {value.step})",
            f"{prefix}{{",
        ]
        for statement in value.body:
            lines.extend(_statement(statement, indent + 1))
        lines.append(f"{prefix}}}")
        return lines
    if isinstance(value, If):
        lines = [f"{prefix}if ({_expression(value.condition)})", f"{prefix}{{"]
        for statement in value.body:
            lines.extend(_statement(statement, indent + 1))
        lines.append(f"{prefix}}}")
        if value.else_body:
            lines.extend((f"{prefix}else", f"{prefix}{{"))
            for statement in value.else_body:
                lines.extend(_statement(statement, indent + 1))
            lines.append(f"{prefix}}}")
        return lines
    raise ShaderTypeError(f"GLSL backend cannot emit {type(value).__name__}")


def emit_glsl(module, declaration=False):
    if isinstance(module, GraphicsModule):
        return _emit_graphics(module)
    if isinstance(module, FunctionModule):
        parameters = ", ".join(
            f"{parameter.qualifier + ' ' if parameter.qualifier != 'in' else ''}"
            f"{parameter.type_name} {_identifier(parameter.name)}"
            for parameter in module.parameters
        )
        if declaration:
            return f"{module.return_type} {module.name}({parameters});\n"
        shared_values = [
            statement for statement in module.statements
            if isinstance(statement, Let)
            and statement.type_name.startswith("shared:")
        ]
        lines = [
            *(
                f"shared {value.type_name.split(':', 1)[1]} "
                f"{_identifier(value.name)};"
                for value in shared_values
            ),
            f"{module.return_type} {module.name}({parameters})",
            "{",
        ]
        for statement in module.statements:
            lines.extend(_statement(statement))
        lines.extend(("}", ""))
        return "\n".join(lines)
    x, y, z = module.workgroup_size
    lines = [
        "#version 460",
    ]
    if any(isinstance(resource.type, AccelerationStructure) for resource in module.resources):
        lines.append("#extension GL_EXT_ray_query : require")
    if any(isinstance(resource.type, SampledTexture2DArray) for resource in module.resources):
        lines.append("#extension GL_EXT_nonuniform_qualifier : require")
    if "subgroup_ballot" in module.capabilities:
        lines.append("#extension GL_KHR_shader_subgroup_basic : require")
        lines.append("#extension GL_KHR_shader_subgroup_ballot : require")
    if "shader_reorder" in module.capabilities:
        lines.append("#extension GL_NV_shader_invocation_reorder : require")
    lines.extend([
        "",
        f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;",
        "",
    ])
    shared_values = [
        statement for statement in module.statements
        if isinstance(statement, Let) and statement.type_name.startswith("shared:")
    ]
    for value in shared_values:
        lines.append(
            f"shared {value.type_name.split(':', 1)[1]} {_identifier(value.name)};"
        )
    if shared_values:
        lines.append("")
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
        struct_type = getattr(
            resource.type, "element_type", getattr(resource.type, "struct_type", None)
        )
        def register_structure(candidate, include_self=True):
            if not isinstance(candidate, StructType):
                return
            for field in candidate.fields:
                nested = field.type.element_type if isinstance(field.type, (RuntimeArrayType, FixedArrayType)) else field.type
                register_structure(nested)
            if include_self:
                structures[candidate.name] = candidate
        register_structure(struct_type, not isinstance(resource.type, StorageRecord))
    for struct_type in structures.values():
        # Storage records are emitted inline as interface blocks below. GLSL
        # does not permit their unsized runtime-array tail in a standalone
        # structure declaration.
        if any(isinstance(field.type, RuntimeArrayType) for field in struct_type.fields):
            continue
        lines.append(f"struct {struct_type.name}")
        lines.append("{")
        lines.extend(
            f"    {_field_declaration(field)}"
            for field in struct_type.fields
        )
        lines.extend(("};", ""))
    for resource in module.resources:
        if isinstance(resource.type, AccelerationStructure):
            lines.append(
                f"layout(set = {resource.set}, binding = {resource.binding}) "
                f"uniform accelerationStructureEXT {_identifier(resource.name)};"
            )
        elif isinstance(resource.type, SampledTexture3DArray):
            lines.append(
                f"layout(set = {resource.set}, binding = {resource.binding}) "
                f"uniform sampler3D {_identifier(resource.name)}"
                f"[{resource.type.count}];"
            )
        elif isinstance(resource.type, SampledTexture2DArray):
            lines.append(
                f"layout(set = {resource.set}, binding = {resource.binding}) "
                f"uniform sampler2D {_identifier(resource.name)}"
                f"[{resource.type.count}];"
            )
        elif isinstance(resource.type, (SampledDepthTexture2D, SampledTexture2D)):
            lines.append(
                f"layout(set = {resource.set}, binding = {resource.binding}) "
                f"uniform texture2D {_identifier(resource.name)};"
            )
        elif isinstance(resource.type, (ComparisonSampler, Sampler)):
            lines.append(
                f"layout(set = {resource.set}, binding = {resource.binding}) "
                f"uniform sampler {_identifier(resource.name)};"
            )
        elif isinstance(resource.type, StorageImageArray):
            access = {
                "read": "readonly ", "write": "writeonly ", "read_write": "",
            }[resource.type.access]
            layout = (
                f", {resource.type.format}"
                if resource.type.format != "unformatted" else ""
            )
            lines.append(
                f"layout(set = {resource.set}, binding = {resource.binding}{layout}) "
                f"uniform {access}image2D {_identifier(resource.name)}"
                f"[{resource.type.count}];"
            )
        elif isinstance(resource.type, StorageImage):
            access = {
                "read": "readonly ", "write": "writeonly ", "read_write": "",
            }[resource.type.access]
            image_type = "image2D"
            if resource.type.format.endswith("ui"):
                image_type = "uimage2D"
            elif resource.type.format.endswith("i"):
                image_type = "iimage2D"
            layout = (
                f", {resource.type.format}"
                if resource.type.format != "unformatted" else ""
            )
            lines.append(
                f"layout(set = {resource.set}, binding = {resource.binding}{layout}) "
                f"uniform {access}{image_type} "
                f"{_identifier(resource.name)};"
            )
        elif isinstance(resource.type, StorageBuffer):
            access = {
                "read": "readonly ",
                "write": "writeonly ",
                "read_write": "",
            }[resource.type.access]
            lines.append(
                f"layout(std430, set = {resource.set}, binding = {resource.binding}) "
                f"{access}buffer {_identifier(resource.name)}_Block"
            )
            lines.extend((
                "{",
                f"    {resource.type.element_type.name} "
                f"{_identifier(resource.name)}[];",
                "};",
            ))
        elif isinstance(resource.type, StorageRecord):
            access = {
                "read": "readonly ",
                "write": "writeonly ",
                "read_write": "",
            }[resource.type.access]
            lines.append(
                f"layout(std430, set = {resource.set}, binding = {resource.binding}) "
                f"{access}buffer {_identifier(resource.name)}_Block"
            )
            lines.append("{")
            lines.extend(
                f"    {_field_declaration(field)}"
                for field in resource.type.struct_type.fields
            )
            lines.append(f"}} {_identifier(resource.name)};")
        elif isinstance(resource.type, UniformBuffer):
            lines.append(
                f"layout(std140, set = {resource.set}, binding = {resource.binding}) "
                f"uniform {_identifier(resource.name)}_Block"
            )
            lines.append("{")
            lines.extend(
                f"    {field.type.name} {_identifier(field.name)};"
                for field in resource.type.struct_type.fields
            )
            lines.extend((f"}} {_identifier(resource.name)};",))
        elif isinstance(resource.type, PushConstants):
            lines.append(
                f"layout(push_constant) uniform {_identifier(resource.name)}_Block"
            )
            lines.append("{")
            lines.extend(
                f"    {field.type.name} {_identifier(field.name)};"
                for field in resource.type.struct_type.fields
            )
            lines.extend((f"}} {_identifier(resource.name)};",))
    if module.functions:
        lines.append("")
        for helper in module.functions:
            parameters = ", ".join(
                f"{parameter.qualifier + ' ' if parameter.qualifier != 'in' else ''}"
                f"{parameter.type_name} {_identifier(parameter.name)}"
                for parameter in helper.parameters
            )
            lines.append(
                f"{helper.return_type} {helper.name}({parameters});"
            )
    if module.externals:
        lines.append("")
        lines.extend(
            emit_glsl(external, declaration=True).rstrip()
            for external in module.externals
        )
    for helper in module.functions:
        lines.append("")
        lines.extend(emit_glsl(helper).rstrip().splitlines())
    lines.extend(("", "void main()", "{"))
    for statement in module.statements:
        lines.extend(_statement(statement))
    lines.extend(("}", ""))
    return "\n".join(lines)


def _glsl_builtin(name):
    return {
        "position": "gl_Position", "vertex_index": "gl_VertexIndex",
        "instance_index": "gl_InstanceIndex", "front_facing": "gl_FrontFacing",
        "frag_depth": "gl_FragDepth",
    }[name]


def _emit_graphics(module):
    lines = ["#version 460", ""]
    declared = set()
    for structure in module.structures:
        if structure.name in declared: continue
        declared.add(structure.name)
        lines.extend((f"struct {structure.name}", "{"))
        lines.extend(f"    {_field_declaration(field)}" for field in structure.fields)
        lines.extend(("};", ""))
    for resource in module.resources:
        if isinstance(resource.type, (SampledDepthTexture2D, SampledTexture2D)):
            lines.append(
                f"layout(set = {resource.set}, binding = {resource.binding}) "
                f"uniform texture2D {_identifier(resource.name)};"
            )
        elif isinstance(resource.type, (ComparisonSampler, Sampler)):
            lines.append(
                f"layout(set = {resource.set}, binding = {resource.binding}) "
                f"uniform sampler {_identifier(resource.name)};"
            )
        elif isinstance(resource.type, UniformBuffer):
            lines.append(f"layout(std140, set = {resource.set}, binding = {resource.binding}) uniform {_identifier(resource.name)}_Block")
            lines.append("{")
            lines.extend(f"    {field.type.name} {_identifier(field.name)};" for field in resource.type.struct_type.fields)
            lines.extend((f"}} {_identifier(resource.name)};", ""))
        elif isinstance(resource.type, StorageBuffer):
            qualifier = "readonly " if resource.type.access == "read" else ""
            lines.extend((f"layout(std430, set = {resource.set}, binding = {resource.binding}) {qualifier}buffer {_identifier(resource.name)}_Block", "{", f"    {resource.type.element_type.name} {_identifier(resource.name)}[];", "};", ""))
        elif isinstance(resource.type, StorageRecord):
            qualifier = "readonly " if resource.type.access == "read" else ""
            lines.extend((f"layout(std430, set = {resource.set}, binding = {resource.binding}) {qualifier}buffer {_identifier(resource.name)}_Block", "{"))
            lines.extend(f"    {_field_declaration(field)}" for field in resource.type.struct_type.fields)
            lines.extend((f"}} {_identifier(resource.name)};", ""))
    if module.output_structure is not None:
        lines.extend((f"struct {module.output_structure.name}", "{"))
        lines.extend(
            f"    {field.type.name} {_identifier(field.name)};"
            for field in module.output_structure.fields
        )
        lines.extend(("};", ""))
    for helper in module.functions:
        lines.extend(emit_glsl(helper).rstrip().splitlines())
        lines.append("")
    arguments = []
    for item in module.inputs:
        if item.location is not None:
            symbol = f"_osh_in_{item.name}"
            lines.append(f"layout(location = {item.location}) in {item.type_name} {symbol};")
            arguments.append(symbol)
        else:
            arguments.append(_glsl_builtin(item.builtin))
    for item in module.outputs:
        if item.location is not None:
            lines.append(f"layout(location = {item.location}) out {item.type_name} _osh_out_{item.name};")
    lines.append("")
    lines.extend(emit_glsl(module.function).rstrip().splitlines())
    lines.extend(("", "void main()", "{"))
    call = f"{module.function.name}({', '.join(arguments)})"
    if module.output_structure is None:
        item = module.outputs[0]
        target = _glsl_builtin(item.builtin) if item.builtin else f"_osh_out_{item.name}"
        lines.append(f"    {target} = {call};")
    else:
        lines.append(f"    {module.output_structure.name} _osh_result = {call};")
        for item in module.outputs:
            target = _glsl_builtin(item.builtin) if item.builtin else f"_osh_out_{item.name}"
            lines.append(f"    {target} = _osh_result.{_identifier(item.name)};")
    lines.extend(("}", ""))
    return "\n".join(lines)
