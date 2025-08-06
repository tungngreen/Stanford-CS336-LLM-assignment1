import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_current_target()
torch_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@triton.jit
def add_kernel(
    x_ptr, # Pointer to the first input vector
    y_ptr, # Pointer to the second input vector
    output_ptr, # Pointer to the output vector
    n_elements, # Number of elements in the vectors
    BLOCK_SIZE: tl.constexpr = 64 # Size of the block to process.
):
    x_ptr = tl.cast(x_ptr, tl.pointer_type(tl.float32))  # Ensure x_ptr is treated as float32
    y_ptr = tl.cast(y_ptr, tl.pointer_type(tl.float32))  # Ensure y_ptr is treated as float32
    output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32)) 
    # tl.device_print("x_ptr: ", x_ptr)
        
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # tl.device_print("pid", pid)
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    tl.device_print("mask", mask)
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
    
def add(
    x: torch.Tensor,
    y: torch.Tensor,
):
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.device == torch_DEVICE and y.device == torch_DEVICE, "Input tensors must be on the same device as Triton kernel"
    # numel provides the total number of elements in the tensor.
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Launch the kernel with a grid size that covers all elements in the input tensors.
    # This is an anonymous function that takes a single argument called meta.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    
    add_kernel[grid](
        x_ptr=x.data_ptr(),
        y_ptr=y.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=64
    )
    
    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda', dtype=torch.float32)
y = torch.rand(size, device='cuda', dtype=torch.float32)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=torch_DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=torch_DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# benchmark.run(print_data=True, show_plots=True)