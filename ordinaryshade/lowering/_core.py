"""Lower the supported Python AST subset into Ordinary Shade IR."""

from __future__ import annotations

import ast
import inspect
import textwrap

from ..entrypoints import ComputeShader, ExternalFunction, GraphicsShader, ShaderFunction
from ..errors import ShaderSyntaxError, ShaderTypeError
from ..ir import (
    Assign, Attribute, Binary, Break, Call, Compare, ComputeModule, Conditional, Continue,
    ExpressionStatement, ForRange, If, Let, Literal, Name, Parameter, Resource,
    Return, Subscript, Unary, While,
    FunctionModule, GraphicsModule, StageInterface,
)
from ..types import (
    AccelerationStructure, ComparisonSampler, FixedArrayType, PushConstants, RuntimeArrayType, ShaderType, StorageBuffer, StorageImage, StorageImageArray, SampledDepthTexture2D, SampledTexture2D, SampledTexture2DArray, SampledTexture3DArray, Sampler, StorageRecord, StructType,
    UniformBuffer, QualifiedType, StageIOType,
)


_BINARY_OPERATORS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%",
    ast.Pow: "**", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.LShift: "<<", ast.RShift: ">>",
}

_COMPARISON_OPERATORS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=",
}

_UNARY_OPERATORS = {
    ast.UAdd: "+", ast.USub: "-", ast.Invert: "~", ast.Not: "!",
}


def _location(node):
    return f"line {getattr(node, 'lineno', '?')}"


class _Lowerer:
    def __init__(self, value_types=None, structures=None, functions=None, capabilities=()):
        self.value_types = dict(value_types or {})
        self.structures = dict(structures or {})
        self.functions = dict(functions or {})
        self.capabilities = frozenset(capabilities)

    def expression_type(self, node):
        if isinstance(node, ast.Name):
            try:
                return self.value_types[node.id]
            except KeyError as error:
                raise ShaderTypeError(
                    f"unknown shader value {node.id!r} at {_location(node)}"
                ) from error
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, int):
                return "int"
            if isinstance(node.value, float):
                return "float"
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in {"osh", "ordinaryshade"}
                and node.value.attr in {
                    "global_invocation_id", "local_invocation_id",
                    "workgroup_id", "num_workgroups", "workgroup_size",
                }
            ):
                if node.attr == "xy":
                    return (
                        "ivec2"
                        if node.value.attr == "global_invocation_id"
                        else "uvec2"
                    )
                if node.attr in {"x", "y", "z"}:
                    return "uint"
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in {"osh", "ordinaryshade"}
                and node.attr == "local_invocation_index"
            ):
                return "uint"
            owner_type = self.expression_type(node.value)
            struct_type = self.structures.get(owner_type)
            if struct_type is not None:
                for field in struct_type.fields:
                    if field.name == node.attr:
                        return field.type.name
                raise ShaderTypeError(
                    f"structure {owner_type!r} has no field {node.attr!r}"
                )
            vector_prefix = next(
                (
                    prefix for prefix in ("bvec", "ivec", "uvec", "vec")
                    if owner_type.startswith(prefix)
                ),
                None,
            )
            if vector_prefix is not None and node.attr:
                component_sets = ({"x", "y", "z", "w"}, {"r", "g", "b", "a"})
                if not any(set(node.attr) <= components for components in component_sets):
                    raise ShaderTypeError(
                        f"invalid vector swizzle {node.attr!r} at {_location(node)}"
                    )
                if len(node.attr) == 1:
                    return {
                        "vec": "float", "ivec": "int", "uvec": "uint",
                        "bvec": "bool",
                    }[vector_prefix]
                if len(node.attr) <= 4:
                    return f"{vector_prefix}{len(node.attr)}"
                raise ShaderTypeError(
                    f"vector swizzle cannot exceed four components at {_location(node)}"
                )
        if isinstance(node, ast.Subscript):
            owner_type = self.expression_type(node.value)
            if owner_type.startswith("storage_image_array:"):
                return "storage_image:" + owner_type.split(":", 1)[1]
            if owner_type.startswith("storage_buffer:"):
                return owner_type.split(":", 1)[1]
            if owner_type.startswith("runtime_array:"):
                return owner_type.split(":", 1)[1]
            if owner_type.startswith("fixed_array:"):
                return owner_type.split(":", 2)[1]
            if owner_type.startswith("local_array:"):
                return owner_type.split(":", 2)[1]
            raise ShaderTypeError(
                f"type {owner_type!r} is not indexable at {_location(node)}"
            )
        if isinstance(node, ast.BinOp):
            left = self.expression_type(node.left)
            right = self.expression_type(node.right)
            if isinstance(node.op, ast.Mult):
                if left.startswith("mat") and "vec" in right:
                    return right
                if "vec" in left and right.startswith("mat"):
                    return left
            if "vec" in left or "mat" in left:
                return left
            if "vec" in right or "mat" in right:
                return right
            if "float" in {left, right}:
                return "float"
            return left
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            operand = self.expression_type(node.operand)
            if isinstance(node.op, ast.Not):
                if operand != "bool":
                    raise ShaderTypeError("not requires a bool shader value")
                return "bool"
            return operand
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                if self.expression_type(value) != "bool":
                    raise ShaderTypeError("and/or require bool shader values")
            return "bool"
        if isinstance(node, ast.IfExp):
            if self.expression_type(node.test) != "bool":
                raise ShaderTypeError("conditional expression requires a bool condition")
            body_type = self.expression_type(node.body)
            else_type = self.expression_type(node.orelse)
            if body_type != else_type:
                raise ShaderTypeError(
                    "conditional expression branches must have the same shader type"
                )
            return body_type
        if isinstance(node, ast.Compare):
            operand = self.expression_type(node.left)
            if "vec" in operand:
                return "b" + operand[operand.index("vec"):]
            return "bool"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"osh", "ordinaryshade"}
            ):
                intrinsic = node.func.attr
                if intrinsic.startswith("subgroup_") and "subgroup_ballot" not in self.capabilities:
                    raise ShaderTypeError(
                        "subgroup operations require capabilities=('subgroup_ballot',)"
                    )
                if intrinsic == "reorder_thread" and "shader_reorder" not in self.capabilities:
                    raise ShaderTypeError(
                        "reorder_thread requires capabilities=('shader_reorder',)"
                    )
                constructors = {
                    "f32": "float", "i32": "int", "u32": "uint",
                    "boolean": "bool", "vec2": "vec2", "vec3": "vec3",
                    "vec4": "vec4", "bvec2": "bvec2", "bvec3": "bvec3",
                    "bvec4": "bvec4", "ivec2": "ivec2", "ivec3": "ivec3",
                    "ivec4": "ivec4", "uvec2": "uvec2", "uvec3": "uvec3",
                    "uvec4": "uvec4",
                    "mat3": "mat3", "mat4": "mat4",
                }
                if intrinsic == "local_array":
                    if len(node.args) != 2 or not (
                        isinstance(node.args[0], ast.Attribute)
                        and isinstance(node.args[0].value, ast.Name)
                        and node.args[0].value.id in {"osh", "ordinaryshade"}
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, int)
                        and node.args[1].value > 0
                    ):
                        raise ShaderTypeError(
                            "local_array() requires a shader type and positive constant count"
                        )
                    element_name = node.args[0].attr
                    element_type = constructors.get(element_name)
                    if element_type is None and element_name in self.structures:
                        element_type = element_name
                    if element_type is None:
                        raise ShaderTypeError(
                            f"unsupported local array element type {element_name!r}"
                        )
                    return f"local_array:{element_type}:{node.args[1].value}"
                if intrinsic == "shared":
                    if len(node.args) != 1 or not (
                        isinstance(node.args[0], ast.Attribute)
                        and isinstance(node.args[0].value, ast.Name)
                        and node.args[0].value.id in {"osh", "ordinaryshade"}
                    ):
                        raise ShaderTypeError("shared() requires a shader type")
                    element_name = node.args[0].attr
                    element_type = constructors.get(element_name)
                    if element_type is None and element_name in self.structures:
                        element_type = element_name
                    if element_type is None:
                        raise ShaderTypeError(
                            f"unsupported shared value type {element_name!r}"
                        )
                    return f"shared:{element_type}"
                if intrinsic in constructors:
                    return constructors[intrinsic]
                if intrinsic == "ray_query":
                    return "rayQueryEXT"
                if intrinsic == "dot":
                    return "float"
                if intrinsic == "length":
                    return "float"
                if intrinsic == "cross":
                    return self.expression_type(node.args[0])
                if intrinsic == "any_value" or intrinsic == "subgroup_elect":
                    return "bool"
                if intrinsic == "subgroup_ballot":
                    return "uvec4"
                if intrinsic in {
                    "subgroup_ballot_bit_count",
                    "subgroup_ballot_exclusive_bit_count",
                }:
                    return "uint"
                if intrinsic in {"subgroup_broadcast_first", "atomic_add"}:
                    return self.expression_type(node.args[0])
                if intrinsic in {"workgroup_barrier", "reorder_thread"}:
                    return "void"
                if intrinsic == "pack_unorm4x8":
                    return "uint"
                if intrinsic == "float_bits_to_uint":
                    return "uint"
                if intrinsic == "uint_bits_to_float":
                    return "float"
                if intrinsic == "bitfield_reverse":
                    return self.expression_type(node.args[0])
                if intrinsic in {"pack_half2x16", "pack_unorm2x16"}:
                    return "uint"
                if intrinsic in {"unpack_half2x16", "unpack_unorm2x16"}:
                    return "vec2"
                if intrinsic == "select":
                    return self.expression_type(node.args[1])
                if intrinsic in {
                    "mix", "minimum", "maximum", "clamp", "normalize", "power",
                    "round", "absolute", "sign", "sqrt", "exp", "exp2", "logarithm", "ceiling",
                    "floor", "log2", "cosine", "sine", "refract",
                    "arctangent2", "arccosine", "fraction",
                }:
                    return self.expression_type(node.args[0])
            owner_type = self.expression_type(node.func.value)
            if owner_type.startswith("storage_image:") and node.func.attr == "load":
                format_name = owner_type.split(":", 1)[1]
                if format_name.endswith("ui"):
                    return "uvec4"
                if format_name.endswith("i"):
                    return "ivec4"
                return "vec4"
            if owner_type.startswith("storage_image:") and node.func.attr == "size":
                if node.args:
                    raise ShaderTypeError("storage image size() takes no arguments")
                return "ivec2"
            if owner_type == "sampled_texture_3d_array" and node.func.attr == "sample":
                if len(node.args) != 2:
                    raise ShaderTypeError(
                        "sampled 3D texture array sample() requires index and coordinate"
                    )
                return "float"
            if owner_type == "sampled_texture_2d_array":
                if node.func.attr == "sample_lod":
                    if len(node.args) != 3:
                        raise ShaderTypeError(
                            "sampled 2D texture array sample_lod() requires index, coordinate, and lod"
                        )
                    return "vec4"
                if node.func.attr == "size":
                    if len(node.args) != 2:
                        raise ShaderTypeError(
                            "sampled 2D texture array size() requires index and lod"
                        )
                    return "ivec2"
                if node.func.attr == "levels":
                    if len(node.args) != 1:
                        raise ShaderTypeError(
                            "sampled 2D texture array levels() requires an index"
                        )
                    return "int"
            if owner_type == "sampled_texture_2d" and node.func.attr == "sample_with":
                if len(node.args) != 2:
                    raise ShaderTypeError(
                        "sampled 2D texture sample_with() requires sampler and coordinate"
                    )
                return "vec4"
            if owner_type == "sampled_texture_2d" and node.func.attr == "sample_level_with":
                if len(node.args) != 3:
                    raise ShaderTypeError(
                        "sampled 2D texture sample_level_with() requires sampler, coordinate, and level"
                    )
                return "vec4"
            if owner_type == "sampled_depth_texture_2d" and node.func.attr == "sample_depth_with":
                if len(node.args) != 2:
                    raise ShaderTypeError(
                        "sampled depth texture sample_depth_with() requires sampler and coordinate"
                    )
                return "float"
            if owner_type == "sampled_depth_texture_2d" and node.func.attr == "sample_depth_level_with":
                if len(node.args) != 3:
                    raise ShaderTypeError(
                        "sampled depth texture sample_depth_level_with() requires sampler, coordinate, and level"
                    )
                return "float"
            if owner_type == "sampled_depth_texture_2d" and node.func.attr == "sample_compare_with":
                if len(node.args) != 3:
                    raise ShaderTypeError(
                        "sampled depth texture sample_compare_with() requires comparison sampler, coordinate, and reference"
                    )
                return "float"
            if owner_type == "rayQueryEXT":
                if node.func.attr == "proceed":
                    return "bool"
                if node.func.attr in {
                    "intersection_type", "primitive_index", "instance_custom_index",
                }:
                    return "uint"
                if node.func.attr == "intersection_t":
                    return "float"
                if node.func.attr == "barycentrics":
                    return "vec2"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in self.structures:
                return node.func.id
            try:
                return self.functions[node.func.id]
            except KeyError:
                pass
        raise ShaderTypeError(
            f"cannot infer shader type for {type(node).__name__} at {_location(node)}"
        )

    def expression(self, node):
        if isinstance(node, ast.Name):
            return Name(node.id)
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, bool)
        ):
            return Literal(node.value)
        if isinstance(node, ast.Attribute):
            return Attribute(self.expression(node.value), node.attr)
        if isinstance(node, ast.Subscript):
            return Subscript(self.expression(node.value), self.expression(node.slice))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return Binary(
                self.expression(node.left), _BINARY_OPERATORS[type(node.op)],
                self.expression(node.right),
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return Unary(_UNARY_OPERATORS[type(node.op)], self.expression(node.operand))
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            operator = "&&" if isinstance(node.op, ast.And) else "||"
            values = [self.expression(value) for value in node.values]
            result = values[0]
            for value in values[1:]:
                result = Binary(result, operator, value)
            return result
        if isinstance(node, ast.IfExp):
            return Conditional(
                self.expression(node.test), self.expression(node.body),
                self.expression(node.orelse),
            )
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in _COMPARISON_OPERATORS
        ):
            return Compare(
                self.expression(node.left),
                _COMPARISON_OPERATORS[type(node.ops[0])],
                self.expression(node.comparators[0]),
                "vec" in self.expression_type(node.left),
            )
        if isinstance(node, ast.Call) and not node.keywords:
            selector_vector = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"osh", "ordinaryshade"}
                and node.func.attr == "select"
                and len(node.args) == 3
                and "vec" in self.expression_type(node.args[0])
            )
            return Call(
                self.expression(node.func),
                tuple(self.expression(argument) for argument in node.args),
                selector_vector,
            )
        raise ShaderSyntaxError(
            f"unsupported shader expression {type(node).__name__} at {_location(node)}"
        )

    def statement(self, node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id not in self.value_types:
                name = target.id
                type_name = self.expression_type(node.value)
                self.value_types[name] = type_name
                return Let(name, type_name, self.expression(node.value))
            if isinstance(target, (ast.Name, ast.Attribute, ast.Subscript)):
                return Assign(self.expression(target), self.expression(node.value))
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name) or node.orelse:
                raise ShaderSyntaxError(
                    f"shader for loops require one name and no else at {_location(node)}"
                )
            call = node.iter
            if not (
                isinstance(call, ast.Call)
                and (
                    isinstance(call.func, ast.Name)
                    and call.func.id in {"range", "unroll_range"}
                    or isinstance(call.func, ast.Attribute)
                    and call.func.attr == "unroll_range"
                )
                and not call.keywords
                and 1 <= len(call.args) <= 3
            ):
                raise ShaderSyntaxError(
                    f"shader loops require range() or unroll_range() with one to three arguments at {_location(node.iter)}"
                )
            unroll = (
                getattr(call.func, "id", None) == "unroll_range"
                or getattr(call.func, "attr", None) == "unroll_range"
            )
            if unroll and any(
                isinstance(child, ast.Continue)
                for statement in node.body for child in ast.walk(statement)
            ):
                raise ShaderSyntaxError(
                    "unroll_range() loops cannot contain continue"
                )
            if len(call.args) == 1:
                start_node, stop_node, step = ast.Constant(0), call.args[0], 1
            elif len(call.args) == 2:
                start_node, stop_node, step = call.args[0], call.args[1], 1
            else:
                start_node, stop_node, step_node = call.args
                if not (
                    isinstance(step_node, ast.Constant)
                    and isinstance(step_node.value, int)
                ):
                    raise ShaderSyntaxError("shader range step must be a constant integer")
                step = step_node.value
            if step == 0:
                raise ShaderSyntaxError("shader range step must be nonzero")
            for bound in (start_node, stop_node):
                if self.expression_type(bound) not in {"int", "uint"}:
                    raise ShaderTypeError("shader range bounds must be integer values")
            original_types = dict(self.value_types)
            self.value_types[node.target.id] = "int"
            body = self.block(node.body)
            self.value_types = original_types
            if unroll and not all(
                isinstance(bound, ast.Constant)
                and isinstance(bound.value, int)
                for bound in (start_node, stop_node)
            ):
                raise ShaderSyntaxError(
                    "unroll_range() bounds must be constant integers"
                )
            return ForRange(
                node.target.id, self.expression(start_node),
                self.expression(stop_node), step, body, unroll,
            )
        if isinstance(node, ast.If):
            condition_type = self.expression_type(node.test)
            if condition_type != "bool":
                raise ShaderTypeError(
                    f"if condition must be bool, got {condition_type} at {_location(node.test)}"
                )
            original_types = dict(self.value_types)
            body = self.block(node.body)
            self.value_types = dict(original_types)
            else_body = self.block(node.orelse)
            self.value_types = original_types
            return If(self.expression(node.test), body, else_body)
        if isinstance(node, ast.While):
            if node.orelse:
                raise ShaderSyntaxError("shader while loops do not support else")
            if self.expression_type(node.test) != "bool":
                raise ShaderTypeError("shader while condition must be bool")
            return While(self.expression(node.test), self.block(node.body))
        if isinstance(node, ast.Expr):
            # Python function docstrings are bare string expressions in the
            # AST.  They are authoring metadata, not executable shader code.
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str,
            ):
                return None
            return ExpressionStatement(self.expression(node.value))
        if isinstance(node, ast.Return):
            return Return(
                self.expression(node.value) if node.value is not None else None
            )
        if isinstance(node, ast.Continue):
            return Continue()
        if isinstance(node, ast.Break):
            return Break()
        if isinstance(node, ast.Pass):
            return None
        raise ShaderSyntaxError(
            f"unsupported shader statement {type(node).__name__} at {_location(node)}"
        )

    def block(self, nodes):
        return tuple(
            lowered for node in nodes
            if (lowered := self.statement(node)) is not None
        )


def _source_function(shader):
    try:
        source = textwrap.dedent(inspect.getsource(shader.function))
    except (OSError, TypeError) as error:
        raise ShaderSyntaxError(
            "shader source is unavailable; define shader functions in a Python module"
        ) from error
    tree = ast.parse(source)
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)), None
    )
    if function is None:
        raise ShaderSyntaxError("could not locate the decorated shader function")
    if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        raise ShaderSyntaxError("variadic and keyword-only shader parameters are unsupported")

    return function


def lower(shader: ComputeShader, *, helpers=(), externals=()) -> ComputeModule:
    if not isinstance(shader, ComputeShader):
        raise ShaderTypeError("compile() expects a function decorated with @compute")
    function = _source_function(shader)
    resources = []
    used_bindings = set()
    next_binding = 0
    annotations = inspect.get_annotations(shader.function, eval_str=True)
    structures = {}
    for argument in function.args.args:
        declaration = annotations.get(argument.arg)
        if not isinstance(
            declaration,
            (AccelerationStructure, StorageImage, SampledDepthTexture2D, SampledTexture2D, SampledTexture2DArray, SampledTexture3DArray, ComparisonSampler, Sampler, StorageBuffer, StorageRecord, UniformBuffer, PushConstants),
        ):
            raise ShaderTypeError(
                f"parameter {argument.arg!r} must have an Ordinary Shade resource annotation"
            )
        if isinstance(declaration, PushConstants):
            structures[declaration.struct_type.name] = declaration.struct_type
            resources.append(Resource(argument.arg, declaration, 0, None))
            continue
        struct_type = getattr(
            declaration, "element_type", getattr(declaration, "struct_type", None)
        )
        def register_structure(candidate):
            if not isinstance(candidate, StructType) or candidate.name in structures:
                return
            structures[candidate.name] = candidate
            for field in candidate.fields:
                nested = (
                    field.type.element_type
                    if isinstance(field.type, (RuntimeArrayType, FixedArrayType))
                    else field.type
                )
                register_structure(nested)
        register_structure(struct_type)
        binding = declaration.binding
        if binding is None:
            while (declaration.set, next_binding) in used_bindings:
                next_binding += 1
            binding = next_binding
            next_binding += 1
        key = (declaration.set, binding)
        if key in used_bindings:
            raise ShaderTypeError(f"duplicate descriptor set/binding {key}")
        used_bindings.add(key)
        resources.append(Resource(argument.arg, declaration, declaration.set, binding))

    resource_types = {}
    for resource in resources:
        if isinstance(resource.type, AccelerationStructure):
            resource_types[resource.name] = "acceleration_structure"
        elif isinstance(resource.type, StorageImageArray):
            resource_types[resource.name] = (
                f"storage_image_array:{resource.type.format}"
            )
        elif isinstance(resource.type, StorageImage):
            resource_types[resource.name] = (
                f"storage_image:{resource.type.format}"
            )
        elif isinstance(resource.type, SampledTexture3DArray):
            resource_types[resource.name] = "sampled_texture_3d_array"
        elif isinstance(resource.type, SampledTexture2DArray):
            resource_types[resource.name] = "sampled_texture_2d_array"
        elif isinstance(resource.type, SampledDepthTexture2D):
            resource_types[resource.name] = "sampled_depth_texture_2d"
        elif isinstance(resource.type, SampledTexture2D):
            resource_types[resource.name] = "sampled_texture_2d"
        elif isinstance(resource.type, Sampler):
            resource_types[resource.name] = "sampler"
        elif isinstance(resource.type, ComparisonSampler):
            resource_types[resource.name] = "comparison_sampler"
        elif isinstance(resource.type, StorageBuffer):
            element_name = resource.type.element_type.name
            resource_types[resource.name] = (
                f"storage_buffer:{element_name}"
            )
        else:
            resource_types[resource.name] = resource.type.struct_type.name
    helper_types = {}
    external_modules = []
    for external in externals:
        if not isinstance(external, ExternalFunction):
            raise ShaderTypeError("external functions must be decorated with @external")
        module = lower_external(external)
        if module.name in helper_types:
            raise ShaderTypeError("external function names must be unique")
        helper_types[module.name] = module.return_type
        external_modules.append(module)
    for helper in helpers:
        if not isinstance(helper, ShaderFunction):
            raise ShaderTypeError("compute helpers must be decorated with @function")
        return_type = inspect.get_annotations(
            helper.function, eval_str=True
        ).get("return")
        if not isinstance(return_type, (ShaderType, StructType)):
            raise ShaderTypeError("shader helper functions require a shader return type")
        if helper.__name__ in helper_types:
            raise ShaderTypeError("shader helper names must be unique")
        helper_types[helper.__name__] = return_type.name
        for declared in inspect.get_annotations(helper.function, eval_str=True).values():
            register_structure(declared)
    helper_modules = tuple(
        lower_function(
            helper, functions=helper_types, external_types=resource_types,
            structures=structures, capabilities=shader.capabilities,
        ) for helper in helpers
    )
    if len(helper_types) != len(helper_modules) + len(external_modules):
        raise ShaderTypeError("shader helper names must be unique")
    lowerer = _Lowerer(
        resource_types, structures, helper_types, shader.capabilities
    )
    statements = lowerer.block(function.body)
    return ComputeModule(
        shader.function.__name__, shader.workgroup_size, tuple(resources), statements,
        helper_modules, shader.capabilities, tuple(structures.values()),
        tuple(external_modules),
    )


def lower_graphics(shader: GraphicsShader, *, helpers=()) -> GraphicsModule:
    if not isinstance(shader, GraphicsShader):
        raise ShaderTypeError("graphics compilation expects @vertex or @fragment")
    function = _source_function(shader)
    annotations = inspect.get_annotations(shader.function, eval_str=True)
    parameters = []
    inputs = []
    structures = {}
    value_types = {}
    resources = []
    used_bindings = set()
    next_binding = 0
    for argument in function.args.args:
        declared = annotations.get(argument.arg)
        if isinstance(declared, StageIOType):
            parameters.append(Parameter(argument.arg, declared.type.name))
            inputs.append(StageInterface(argument.arg, declared.type.name, declared.location, declared.builtin, declared.invariant))
            value_types[argument.arg] = declared.type.name
            continue
        if not isinstance(declared, (UniformBuffer, StorageBuffer, StorageRecord, SampledDepthTexture2D, SampledTexture2D, ComparisonSampler, Sampler)):
            raise ShaderTypeError(
                f"graphics parameter {argument.arg!r} requires stage IO or a portable buffer resource"
            )
        binding = declared.binding
        if binding is None:
            while (declared.set, next_binding) in used_bindings: next_binding += 1
            binding = next_binding; next_binding += 1
        key = (declared.set, binding)
        if key in used_bindings: raise ShaderTypeError(f"duplicate descriptor set/binding {key}")
        used_bindings.add(key)
        resources.append(Resource(argument.arg, declared, declared.set, binding))
        structure = getattr(declared, "struct_type", getattr(declared, "element_type", None))
        if isinstance(structure, StructType): structures[structure.name] = structure
        if isinstance(declared, SampledDepthTexture2D):
            value_types[argument.arg] = "sampled_depth_texture_2d"
        elif isinstance(declared, SampledTexture2D):
            value_types[argument.arg] = "sampled_texture_2d"
        elif isinstance(declared, Sampler):
            value_types[argument.arg] = "sampler"
        elif isinstance(declared, ComparisonSampler):
            value_types[argument.arg] = "comparison_sampler"
        elif isinstance(declared, StorageBuffer):
            value_types[argument.arg] = f"storage_buffer:{structure.name}"
        else:
            value_types[argument.arg] = structure.name
    declared_return = annotations.get("return")
    outputs = []
    output_structure = None
    if isinstance(declared_return, StageIOType):
        return_type = declared_return.type
        outputs.append(StageInterface("result", return_type.name, declared_return.location, declared_return.builtin, declared_return.invariant))
    elif isinstance(declared_return, StructType):
        output_structure = declared_return
        structures[declared_return.name] = declared_return
        return_type = declared_return
        for field in declared_return.fields:
            if not isinstance(field.type, StageIOType):
                raise ShaderTypeError("graphics output structure fields require location() or builtin()")
            outputs.append(StageInterface(field.name, field.type.type.name, field.type.location, field.type.builtin, field.type.invariant))
    else:
        raise ShaderTypeError("graphics shader return requires stage output annotation")
    # The expression type system sees the underlying value type, while the
    # interface metadata remains attached to the module.
    # Resource-backed structures participate in expression typing too.  Keep
    # them available so a uniform field can be assigned locally and swizzled,
    # not merely embedded directly in a larger expression.
    lower_structures = dict(structures)
    if output_structure is not None:
        from ..types import StructField
        lowered = StructType(output_structure.name, tuple(
            StructField(field.name, field.type.type) for field in output_structure.fields
        ))
        lower_structures[lowered.name] = lowered
        return_type = lowered
        output_structure = lowered
    helper_types = {}
    for helper in helpers:
        if not isinstance(helper, ShaderFunction):
            raise ShaderTypeError(
                "graphics helpers must be decorated with @function"
            )
        annotations = inspect.get_annotations(helper.function, eval_str=True)
        helper_return_type = annotations.get("return")
        if not isinstance(helper_return_type, (ShaderType, StructType)):
            raise ShaderTypeError(
                "shader helper functions require a shader return type"
            )
        if helper.__name__ in helper_types:
            raise ShaderTypeError("shader helper names must be unique")
        helper_types[helper.__name__] = helper_return_type.name
        for declared in annotations.values():
            if isinstance(declared, StructType):
                lower_structures[declared.name] = declared
    helper_modules = tuple(
        lower_function(
            helper, functions=helper_types, structures=lower_structures,
        )
        for helper in helpers
    )
    lowerer = _Lowerer(value_types, lower_structures, helper_types)
    statements = lowerer.block(function.body)
    fn = FunctionModule(shader.function.__name__, tuple(parameters), return_type.name, statements)
    resource_structures = tuple(
        value for name, value in lower_structures.items()
        if output_structure is None or name != output_structure.name
    )
    return GraphicsModule(
        shader.function.__name__, shader.stage, fn, tuple(inputs),
        tuple(outputs), output_structure, tuple(resources),
        resource_structures, helper_modules,
    )


def lower_function(
    shader: ShaderFunction, *, functions=None, external_types=None,
    structures=None, capabilities=(),
) -> FunctionModule:
    if not isinstance(shader, ShaderFunction):
        raise ShaderTypeError("compile_function() expects a function decorated with @function")
    function = _source_function(shader)
    annotations = inspect.get_annotations(shader.function, eval_str=True)
    parameters = []
    for argument in function.args.args:
        declared = annotations.get(argument.arg)
        qualifier = "in"
        if isinstance(declared, QualifiedType):
            qualifier = declared.qualifier
            declared = declared.type
        if not isinstance(declared, (ShaderType, StructType)):
            raise ShaderTypeError(f"parameter {argument.arg!r} requires a shader value type")
        parameters.append(Parameter(argument.arg, declared.name, qualifier))
    return_type = annotations.get("return")
    if not isinstance(return_type, (ShaderType, StructType)):
        raise ShaderTypeError("shader helper functions require a shader return type")
    value_types = dict(external_types or {})
    value_types.update({parameter.name: parameter.type_name for parameter in parameters})
    helper_structures = dict(structures or {})
    for declared in annotations.values():
        if isinstance(declared, StructType):
            helper_structures[declared.name] = declared
    lowerer = _Lowerer(
        value_types, helper_structures, functions=functions,
        capabilities=capabilities,
    )
    statements = lowerer.block(function.body)
    def contains_return(items):
        return any(
            isinstance(statement, Return)
            or isinstance(statement, If) and (
                contains_return(statement.body) or contains_return(statement.else_body)
            )
            for statement in items
        )
    if return_type.name != "void" and not contains_return(statements):
        raise ShaderSyntaxError("shader helper function must return a value")
    return FunctionModule(
        shader.__name__, tuple(parameters), return_type.name, statements,
    )


def lower_external(shader: ExternalFunction) -> FunctionModule:
    annotations = inspect.get_annotations(shader.function, eval_str=True)
    signature = inspect.signature(shader.function)
    parameters = []
    for parameter in signature.parameters.values():
        declared = annotations.get(parameter.name)
        qualifier = "in"
        if isinstance(declared, QualifiedType):
            qualifier = declared.qualifier
            declared = declared.type
        if not isinstance(declared, (ShaderType, StructType)):
            raise ShaderTypeError(
                f"external parameter {parameter.name!r} requires a shader value type"
            )
        parameters.append(Parameter(parameter.name, declared.name, qualifier))
    return_type = annotations.get("return")
    if not isinstance(return_type, (ShaderType, StructType)):
        raise ShaderTypeError("external functions require a shader return type")
    return FunctionModule(
        shader.function.__name__, tuple(parameters), return_type.name, (),
    )
