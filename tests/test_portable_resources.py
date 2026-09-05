"""Target compilation and binding regression coverage for portable resources."""
import shutil

import pytest
import ordinaryshade as osh


@osh.structure
class Parameters:
    gain: osh.f32


@osh.compute()
def volume_sample(
    volume: osh.sampled_texture_3d(binding=0),
    filtering: osh.sampler(binding=1),
    output: osh.storage_buffer(osh.vec4, binding=2),
):
    output[0] = volume.sample_level_with(filtering, osh.vec3(0.5), 0.0)


@osh.compute()
def fallback(
    output: osh.storage_buffer(osh.f32),
    params: osh.push_constants(Parameters, wgsl_binding=0),
):
    output[0] = params.gain


@osh.compute()
def collision(
    params: osh.push_constants(Parameters, wgsl_binding=0),
    output: osh.storage_buffer(osh.f32, binding=0),
):
    output[0] = params.gain


@osh.compute()
def reverse_collision(
    output: osh.storage_buffer(osh.f32, binding=0),
    params: osh.push_constants(Parameters, wgsl_binding=0),
):
    output[0] = params.gain


@osh.compute()
def fallback_collision(
    first: osh.push_constants(Parameters, wgsl_binding=0),
    second: osh.push_constants(Parameters, wgsl_binding=0),
):
    return


@osh.compute()
def other_group(
    params: osh.push_constants(Parameters, wgsl_set=2, wgsl_binding=0),
    output: osh.storage_buffer(osh.f32, binding=0),
):
    output[0] = params.gain


def compile_target(shader, target):
    if target == "spirv" and not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator unavailable")
    return osh.compile(
        shader, target=target,
        validate=target == "wgsl" and bool(shutil.which("naga")),
    )


@pytest.mark.parametrize("target", ["glsl", "wgsl", "spirv"])
def test_compute_volume_sampling(target):
    result = compile_target(volume_sample, target)
    resource = result.reflection.resources[0]
    assert (resource.kind, resource.format, resource.binding) == (
        "sampled_texture_3d", "rgba", 0,
    )


@pytest.mark.parametrize("target", ["glsl", "wgsl", "spirv"])
def test_fallback_binding_and_reflection(target):
    result = compile_target(fallback, target)
    output, params = result.reflection.resources
    assert output.binding == (1 if target == "wgsl" else 0)
    assert (params.kind, params.set, params.binding) == (
        ("uniform_buffer", 0, 0) if target == "wgsl"
        else ("push_constants", 0, -1)
    )


@pytest.mark.parametrize("shader", [collision, reverse_collision, fallback_collision])
def test_fallback_collisions_rejected_without_validator(shader):
    with pytest.raises(osh.ShaderTypeError, match="duplicate descriptor"):
        osh.compile(shader, target="wgsl")


@pytest.mark.parametrize("shader", [collision, reverse_collision])
def test_vulkan_does_not_reserve_wgsl_fallback(shader):
    compile_target(shader, "spirv")


@pytest.mark.parametrize("target", ["glsl", "wgsl", "spirv"])
def test_fallback_group_is_target_specific(target):
    result = compile_target(other_group, target)
    params, output = result.reflection.resources
    assert (params.set, params.binding) == ((2, 0) if target == "wgsl" else (0, -1))
    assert (output.set, output.binding) == (0, 0)
