"""Compute helper libraries need not declare resource parameters."""
import ordinaryshade as osh


@osh.structure
class Value:
    number: osh.f32


@osh.function
def helper(value: Value) -> osh.f32:
    return value.number


@osh.compute()
def library():
    pass


def test_helper_structure_is_registered_without_resources():
    source = osh.compile(library, helpers=(helper,)).source
    assert 'struct Value' in source
    assert 'float helper(Value value)' in source
