import pytest
import ordinaryshade as osh

@osh.compute(workgroup_size=(64,1,1))
def reverse_group(output: osh.storage_buffer(osh.u32,binding=0)):
    shared_values=osh.shared(osh.array(osh.u32,64))
    lane=osh.local_invocation_index
    i=osh.global_invocation_id.x
    shared_values[lane]=i if i<osh.array_length(output) else osh.u32(0)
    osh.workgroup_barrier()
    if i<osh.array_length(output):
        output[i]=shared_values[osh.u32(63)-lane]

@osh.function
def shared_helper(value: osh.u32) -> osh.u32:
    shared_values=osh.shared(osh.array(osh.u32,4))
    shared_values[0]=value
    osh.workgroup_barrier()
    return shared_values[0]

def test_array_declarations_and_hoisting():
    glsl=osh.compile(reverse_group).source
    wgsl=osh.compile(reverse_group,target='wgsl').source
    assert 'shared uint shared_values[64];' in glsl
    assert 'var<workgroup> shared_values: array<u32, 64>;' in wgsl
    assert glsl.index('shared uint')<glsl.index('void main')
    assert 'shared uint shared_values[4];' in osh.compile_function(shared_helper).source
    assert 'var<workgroup> shared_values: array<u32, 4>;' in osh.compile_function(shared_helper,target='wgsl').source

@osh.compute(workgroup_size=(1,1,1))
def empty_array():
    values=osh.shared(osh.array(osh.u32,0))

@osh.compute(workgroup_size=(1,1,1))
def varying_array():
    values=osh.shared(osh.array(osh.u32,osh.local_invocation_index))

@osh.compute(workgroup_size=(1,1,1))
def boolean_size():
    values=osh.shared(osh.array(osh.u32,True))

@pytest.mark.parametrize('shader',(empty_array,varying_array,boolean_size))
def test_invalid_shared_array_sizes(shader):
    with pytest.raises(osh.ShaderTypeError,match='positive constant'):
        osh.compile(shader)
