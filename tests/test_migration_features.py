"""Typed language features needed by transport and portable volume shaders."""
import subprocess
import shutil
import pytest
import ordinaryshade as osh

@osh.function
def output_parameter(value: osh.vec3, result: osh.out(osh.vec3)) -> osh.void:
    result = osh.reflect(value, osh.vec3(0.0, 1.0, 0.0))

@osh.function
def scoped_loop() -> osh.f32:
    result = 0.0
    i = 0
    while i < 2:
        temporary = osh.vec3(1.0)
        result = result + temporary.x
        i = i + 1
    temporary = 2.0
    return result + temporary

@osh.function
def variant(value: osh.f32) -> osh.f32:
    result = value
    if osh.specialization('QUALITY > 1'):
        result = result * 2.0
    else:
        result = result + 1.0
    return result

@osh.function
def invalid_variant() -> osh.f32:
    if osh.specialization('FLAG\n#error injected'):
        return 1.0
    return 0.0

@osh.compute(workgroup_size=(4, 4, 4))
def volume_upload(source: osh.storage_buffer(osh.f32, binding=0),
                  target: osh.storage_image('rgba16f', dimensions=3, access='write', binding=1)):
    index = osh.uvec3(osh.global_invocation_id.x, osh.global_invocation_id.y, osh.global_invocation_id.z)
    size = osh.uvec3(target.size())
    if osh.any_value(index >= size):
        return
    offset = (index.z * size.y + index.y) * size.x + index.x
    if offset < osh.array_length(source):
        target.store(osh.ivec3(index), osh.vec4(source[offset], 0.0, 0.0, 1.0))

@osh.compute()
def numeric_and_atomic(output: osh.storage_buffer(osh.u32)):
    value = osh.unpack_unorm4x8(output[0])
    invalid = osh.any_value(osh.is_nan(value)) or osh.any_value(osh.is_inf(value))
    if invalid:
        osh.atomic_or(output[0], osh.u32(32))
    else:
        output[0] = osh.u32(osh.modulo(value.x, 2.0))


def test_out_and_loop_scope():
    assert 'out vec3 result' in osh.compile_function(output_parameter).source
    assert 'float temporary = 2.0' in osh.compile_function(scoped_loop).source
    with pytest.raises((osh.ShaderTypeError, osh.ShaderSyntaxError)):
        osh.compile_function(output_parameter, target='wgsl')


def test_specialization_is_static_and_rejects_directive_injection():
    source = osh.compile_function(variant).source
    assert '#if QUALITY > 1' in source and '#else' in source and '#endif' in source
    with pytest.raises((osh.ShaderTypeError, osh.ShaderSyntaxError)):
        osh.compile_function(variant, target='wgsl')
    with pytest.raises((osh.ShaderTypeError, osh.ShaderSyntaxError)):
        osh.compile_function(invalid_variant)


def test_volume_dimensions_and_runtime_array_length():
    glsl = osh.compile(volume_upload).source
    wgsl = osh.compile(volume_upload, target='wgsl').source
    assert 'image3D target' in glsl and 'uint(source.length())' in glsl
    assert 'texture_storage_3d<rgba16float, write>' in wgsl
    assert 'arrayLength(&source)' in wgsl
    with pytest.raises((ValueError, osh.ShaderTypeError)):
        osh.storage_image('rgba16f', dimensions=4)


@pytest.mark.parametrize('shader', [volume_upload, numeric_and_atomic])
def test_new_features_compile_to_spirv(shader, tmp_path):
    compiler = shutil.which('glslangValidator')
    if compiler is None:
        pytest.skip('glslangValidator unavailable')
    source = tmp_path / 'test.comp'
    source.write_text(osh.compile(shader).source)
    result = subprocess.run([compiler, '-V', '--target-env', 'vulkan1.2', str(source), '-o', str(tmp_path / 'test.spv')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

@osh.compute()
def atomic_bits(output: osh.storage_buffer(osh.u32)):
    osh.atomic_or(output[0], osh.u32(1))


def test_atomic_or_rejects_unsupported_wgsl_storage_atomic_abi():
    with pytest.raises(osh.ShaderTypeError, match='atomic_or'):
        osh.compile(atomic_bits, target='wgsl')

@osh.compute()
def vector_bits(source: osh.storage_buffer(osh.vec4, binding=0),
                output: osh.storage_buffer(osh.uvec4, binding=1),
                roundtrip: osh.storage_buffer(osh.vec4, binding=2)):
    bits = osh.float_bits_to_uint(source[0])
    output[0] = bits
    value = osh.uint_bits_to_float(bits)
    roundtrip[0] = value


def test_vector_bitcasts_preserve_component_count(tmp_path):
    glsl = osh.compile(vector_bits).source
    wgsl = osh.compile(vector_bits, target='wgsl').source
    assert 'uvec4 bits = floatBitsToUint(source[0]);' in glsl
    assert 'vec4 value = uintBitsToFloat(bits);' in glsl
    assert 'bitcast<vec4<u32>>(source[0])' in wgsl
    assert 'bitcast<vec4<f32>>(bits)' in wgsl
    test_new_features_compile_to_spirv(vector_bits, tmp_path)
