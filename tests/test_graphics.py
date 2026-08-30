import shutil

import ordinaryshade as osh


@osh.structure
class VertexOutput:
    position: osh.builtin(osh.vec4, "position")
    color: osh.location(osh.vec3, 0)


@osh.structure
class CameraUniforms:
    tint: osh.vec3
    scale: osh.f32


@osh.vertex
def triangle_vertex(position: osh.location(osh.vec2, 0)) -> osh.builtin(osh.vec4, "position"):
    return osh.vec4(position, 0.0, 1.0)


@osh.fragment
def triangle_fragment() -> osh.location(osh.vec4, 0):
    return osh.vec4(0.95, 0.45, 0.15, 1.0)


@osh.function
def tint_surface(color: osh.vec3, amount: osh.f32) -> osh.vec3:
    """Apply a scalar tint while retaining Python-side documentation."""
    return color * amount


@osh.function(name="stable_tint_hook")
def locally_named_tint(color: osh.vec3) -> osh.vec3:
    return color * 0.75


@osh.fragment
def helper_fragment(color: osh.location(osh.vec3, 0)) -> osh.location(osh.vec4, 0):
    return osh.vec4(tint_surface(color, 0.5), 1.0)


@osh.vertex
def varying_vertex(position: osh.location(osh.vec2, 0)) -> VertexOutput:
    return VertexOutput(osh.vec4(position, 0.0, 1.0), osh.vec3(1.0, 0.5, 0.2))


@osh.fragment
def varying_fragment(color: osh.location(osh.vec3, 0)) -> osh.location(osh.vec4, 0):
    return osh.vec4(color, 1.0)


@osh.fragment
def uniform_fragment(
    color: osh.location(osh.vec3, 0),
    camera: osh.uniform_buffer(CameraUniforms, binding=2),
) -> osh.location(osh.vec4, 0):
    return osh.vec4(color * camera.tint * camera.scale, 1.0)


@osh.fragment
def assigned_uniform_field_fragment(
    color: osh.location(osh.vec3, 0),
    camera: osh.uniform_buffer(CameraUniforms, binding=2),
) -> osh.location(osh.vec4, 0):
    tint = camera.tint
    red_green = tint.xy
    return osh.vec4(color * osh.vec3(red_green, tint.z), 1.0)


@osh.fragment
def textured_fragment(
    uv: osh.location(osh.vec2, 0),
    image: osh.sampled_texture_2d(binding=3),
    filtering: osh.sampler(binding=4),
) -> osh.location(osh.vec4, 0):
    return image.sample_with(filtering, uv)


@osh.fragment
def depth_textured_fragment(
    uv: osh.location(osh.vec2, 0),
    image: osh.sampled_depth_texture_2d(binding=3),
    filtering: osh.sampler(binding=4),
) -> osh.location(osh.vec4, 0):
    depth = image.sample_depth_with(filtering, uv)
    return osh.vec4(osh.vec3(depth), 1.0)


@osh.fragment
def compared_depth_fragment(
    uv: osh.location(osh.vec2, 0),
    image: osh.sampled_depth_texture_2d(binding=3),
    comparison: osh.comparison_sampler(binding=4),
) -> osh.location(osh.vec4, 0):
    visibility = image.sample_compare_with(comparison, uv, 0.5)
    return osh.vec4(osh.vec3(visibility), 1.0)


def test_glsl_graphics_stages_and_reflection():
    vertex = osh.compile(triangle_vertex)
    fragment = osh.compile(triangle_fragment)
    assert "layout(location = 0) in vec2 _osh_in_position;" in vertex.source
    assert "gl_Position = triangle_vertex(_osh_in_position);" in vertex.source
    assert "layout(location = 0) out vec4 _osh_out_result;" in fragment.source
    assert vertex.reflection.stage == "vertex"
    assert vertex.reflection.inputs[0].location == 0
    assert vertex.reflection.outputs[0].builtin == "position"


def test_wgsl_graphics_stages_validate_when_naga_is_available():
    validate = shutil.which("naga") is not None
    vertex = osh.compile(triangle_vertex, target="wgsl", validate=validate)
    fragment = osh.compile(triangle_fragment, target="wgsl", validate=validate)
    assert "@vertex" in vertex.source
    assert "@location(0) position: vec2<f32>" in vertex.source
    assert "-> @builtin(position) vec4<f32>" in vertex.source
    assert "@fragment" in fragment.source


def test_graphics_stages_support_portable_helper_functions():
    for target in ("glsl", "wgsl"):
        result = osh.compile(
            helper_fragment, target=target, helpers=(tint_surface,),
        )
        assert "tint_surface" in result.source
        assert "tint_surface(color, 0.5)" in result.source


def test_shader_helpers_can_export_a_stable_embedding_symbol():
    assert locally_named_tint.__name__ == "stable_tint_hook"


def test_graphics_spirv_validates_when_glslang_is_available():
    if shutil.which("glslangValidator") is None:
        return
    assert osh.compile(triangle_vertex, target="spirv").binary[:4] == b"\x03\x02#\x07"
    assert osh.compile(triangle_fragment, target="spirv").binary[:4] == b"\x03\x02#\x07"


def test_cross_stage_varyings_link_for_both_targets():
    for target in ("glsl", "wgsl"):
        vertex = osh.compile(varying_vertex, target=target)
        fragment = osh.compile(varying_fragment, target=target)
        linked = osh.link_graphics(vertex, fragment)
        assert linked.varyings[0].type == "vec3"
        assert linked.varyings[0].location == 0


def test_graphics_uniform_resources_emit_and_reflect_for_both_targets():
    glsl = osh.compile(uniform_fragment, target="glsl")
    wgsl = osh.compile(uniform_fragment, target="wgsl")
    assert "layout(std140, set = 0, binding = 2)" in glsl.source
    assert "@group(0) @binding(2) var<uniform>" in wgsl.source
    assert glsl.reflection.resources[0].kind == "uniform_buffer"
    assert glsl.reflection.resources[0].binding == 2


def test_uniform_fields_can_be_assigned_and_swizzled():
    for target in ("glsl", "wgsl"):
        result = osh.compile(assigned_uniform_field_fragment, target=target)
        assert "tint" in result.source


def test_separate_texture_and_sampler_emit_for_both_graphics_targets():
    glsl = osh.compile(textured_fragment, target="glsl")
    wgsl = osh.compile(textured_fragment, target="wgsl")
    assert "uniform texture2D image;" in glsl.source
    assert "uniform sampler filtering;" in glsl.source
    assert "texture(sampler2D(image, filtering), uv)" in glsl.source
    assert "var image: texture_2d<f32>;" in wgsl.source
    assert "var filtering: sampler;" in wgsl.source
    assert "textureSample(image, filtering, uv)" in wgsl.source
    assert [item.kind for item in glsl.reflection.resources] == [
        "sampled_texture_2d", "sampler",
    ]


def test_depth_texture_emits_scalar_sampling_for_both_graphics_targets():
    glsl = osh.compile(depth_textured_fragment, target="glsl")
    validate = shutil.which("naga") is not None
    wgsl = osh.compile(depth_textured_fragment, target="wgsl", validate=validate)
    assert "uniform texture2D image;" in glsl.source
    assert "texture(sampler2D(image, filtering), uv).r" in glsl.source
    assert "var image: texture_depth_2d;" in wgsl.source
    assert "textureSample(image, filtering, uv)" in wgsl.source
    assert [item.kind for item in glsl.reflection.resources] == [
        "sampled_depth_texture_2d", "sampler",
    ]


def test_depth_comparison_sampler_emits_for_both_graphics_targets():
    glsl = osh.compile(compared_depth_fragment, target="glsl")
    validate = shutil.which("naga") is not None
    wgsl = osh.compile(compared_depth_fragment, target="wgsl", validate=validate)
    assert "sampler2DShadow(image, comparison)" in glsl.source
    assert "var comparison: sampler_comparison;" in wgsl.source
    assert "textureSampleCompare(image, comparison, uv, 0.5)" in wgsl.source
    assert [item.kind for item in glsl.reflection.resources] == [
        "sampled_depth_texture_2d", "comparison_sampler",
    ]


def test_graphics_structured_storage_buffers_are_indexable():
    @osh.structure
    class Surface:
        color: osh.vec4

    @osh.fragment
    def material_fragment(
        index: osh.location(osh.f32, 0),
        surfaces: osh.storage_buffer(Surface, access="read", binding=5),
    ) -> osh.location(osh.vec4, 0):
        color = surfaces[osh.u32(index)].color
        return color

    validate = shutil.which("naga") is not None
    glsl = osh.compile(material_fragment, target="glsl")
    wgsl = osh.compile(material_fragment, target="wgsl", validate=validate)
    assert "readonly buffer surfaces_Block" in glsl.source
    assert "surfaces[uint(index)].color" in glsl.source
    assert "var<storage, read> surfaces: array<Surface>" in wgsl.source
    assert "surfaces[u32(index)].color" in wgsl.source
