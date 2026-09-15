import ordinaryshade as osh
import pytest

@osh.compute(workgroup_size=(64,1,1),capabilities=('buffer_float32_atomic_add',))
def accumulate(sums: osh.storage_buffer(osh.f32,access='read_write',binding=0)):
    previous=osh.atomic_add(sums[0],0.125)

@osh.compute(workgroup_size=(64,1,1))
def missing_capability(sums: osh.storage_buffer(osh.f32,access='read_write',binding=0)):
    previous=osh.atomic_add(sums[0],0.125)

@osh.compute(workgroup_size=(64,1,1),capabilities=('buffer_float32_atomic_add',))
def wrong_type(sums: osh.storage_buffer(osh.f32,access='read_write',binding=0)):
    previous=osh.atomic_add(sums[0],osh.u32(1))

def test_glsl_emits_float_atomic_extension():
    source=osh.compile(accumulate).source
    assert '#extension GL_EXT_shader_atomic_float : require' in source
    assert 'atomicAdd(' in source

def test_float_atomic_requires_capability():
    with pytest.raises(osh.ShaderTypeError,match='capability'):
        osh.compile(missing_capability)

def test_float_atomic_requires_matching_type():
    with pytest.raises(osh.ShaderTypeError,match='matching scalar float'):
        osh.compile(wrong_type)

def test_wgsl_rejects_float_atomic_capability():
    with pytest.raises(osh.ShaderTypeError,match='WGSL buffer float32'):
        osh.compile(accumulate,target='wgsl')

@osh.compute(workgroup_size=(64,1,1))
def bare_missing(sums: osh.storage_buffer(osh.f32,access='read_write',binding=0)):
    osh.atomic_add(sums[0],0.125)

@osh.compute(workgroup_size=(64,1,1),capabilities=('buffer_float32_atomic_add',))
def shared_float():
    value=osh.shared(osh.f32)
    osh.atomic_add(value,0.125)

def test_bare_call_checks_capability():
    with pytest.raises(osh.ShaderTypeError,match='capability'):
        osh.compile(bare_missing)

def test_buffer_capability_does_not_enable_shared_atomics():
    with pytest.raises(osh.ShaderTypeError,match='storage-buffer'):
        osh.compile(shared_float)
