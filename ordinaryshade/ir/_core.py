"""Backend-neutral intermediate representation for Ordinary Shade."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import PushConstants, SampledTexture2DArray, SampledTexture3DArray, StorageBuffer, StorageImage, StorageRecord, UniformBuffer


@dataclass(frozen=True, slots=True)
class Resource:
    name: str
    type: StorageImage | SampledTexture2DArray | SampledTexture3DArray | StorageBuffer | StorageRecord | UniformBuffer | PushConstants
    set: int
    binding: int | None


class Expression:
    pass


@dataclass(frozen=True, slots=True)
class Name(Expression):
    value: str


@dataclass(frozen=True, slots=True)
class Literal(Expression):
    value: int | float | bool


@dataclass(frozen=True, slots=True)
class Attribute(Expression):
    value: Expression
    attribute: str


@dataclass(frozen=True, slots=True)
class Subscript(Expression):
    value: Expression
    index: Expression


@dataclass(frozen=True, slots=True)
class Binary(Expression):
    left: Expression
    operator: str
    right: Expression


@dataclass(frozen=True, slots=True)
class Unary(Expression):
    operator: str
    value: Expression


@dataclass(frozen=True, slots=True)
class Conditional(Expression):
    condition: Expression
    when_true: Expression
    when_false: Expression


@dataclass(frozen=True, slots=True)
class Compare(Expression):
    left: Expression
    operator: str
    right: Expression
    vector: bool = False


@dataclass(frozen=True, slots=True)
class Call(Expression):
    function: Expression
    arguments: tuple[Expression, ...]
    selector_vector: bool = False


class Statement:
    pass


@dataclass(frozen=True, slots=True)
class Let(Statement):
    name: str
    type_name: str
    value: Expression


@dataclass(frozen=True, slots=True)
class ExpressionStatement(Statement):
    value: Expression


@dataclass(frozen=True, slots=True)
class Return(Statement):
    value: Expression | None = None


@dataclass(frozen=True, slots=True)
class Continue(Statement):
    pass


@dataclass(frozen=True, slots=True)
class Break(Statement):
    pass


@dataclass(frozen=True, slots=True)
class While(Statement):
    condition: Expression
    body: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class If(Statement):
    condition: Expression
    body: tuple[Statement, ...]
    else_body: tuple[Statement, ...] = ()


@dataclass(frozen=True, slots=True)
class Assign(Statement):
    target: Expression
    value: Expression


@dataclass(frozen=True, slots=True)
class ForRange(Statement):
    variable: str
    start: Expression
    stop: Expression
    step: int
    body: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class Parameter:
    name: str
    type_name: str
    qualifier: str = "in"


@dataclass(frozen=True, slots=True)
class ComputeModule:
    name: str
    workgroup_size: tuple[int, int, int]
    resources: tuple[Resource, ...]
    statements: tuple[Statement, ...]
    functions: tuple[FunctionModule, ...] = ()
    capabilities: tuple[str, ...] = ()
    structures: tuple[object, ...] = ()
    externals: tuple[FunctionModule, ...] = ()


@dataclass(frozen=True, slots=True)
class FunctionModule:
    name: str
    parameters: tuple[Parameter, ...]
    return_type: str
    statements: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class StageInterface:
    name: str
    type_name: str
    location: int | None = None
    builtin: str | None = None


@dataclass(frozen=True, slots=True)
class GraphicsModule:
    name: str
    stage: str
    function: FunctionModule
    inputs: tuple[StageInterface, ...]
    outputs: tuple[StageInterface, ...]
    output_structure: object | None = None
    resources: tuple[Resource, ...] = ()
    structures: tuple[object, ...] = ()
