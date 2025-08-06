import triton
import triton.language as tl

from einops import rearrange
import torch


@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, # Input pointer for the 'x' tensor (data matrix).
    output_ptr, # Output pointer for the resulting weighted sum vector.
    x_stride_row, x_stride_dim, # Strides for the 'x' tensor: x_stride_row is the distance in memory to move to the next row, and x_stride_dim is the distance to move to the next element within a row.
    weight_stride_dim, # Stride for the 'weight' tensor. For a 1D vector, this is typically 1, meaning elements are contiguous.
    output_stride_row, # Stride for the 'output' tensor. For a 1D vector, this is typically 1.
    ROWS, D, # ROWS is the total number of rows in the 'x' tensor, and D is the total number of columns (or dimensions).
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # Compile-time constants defining the dimensions of the processing tiles. These must be known at compilation for optimization.
):
    # Each Triton program instance (thread block) will compute the weighted sum for a tile of rows from the 'x' tensor.
    # `tl.program_id(0)` returns the ID of the current program instance along the first dimension of the grid.
    # This ID is used to determine which specific tile of rows this instance is responsible for.
    row_tile_idx = tl.program_id(0)

    # Block pointers are a core concept in Triton for efficiently accessing ND (N-dimensional) regions of memory.
    # They allow for selecting a block of data and moving that selection around.
    # To properly construct a block pointer, the following information is required:
    # - The base pointer to the first element of the tensor in global memory.
    # - The overall shape (dimensions) of the tensor, which is crucial for handling out-of-bounds memory accesses during loads/stores.
    # - The strides of each dimension, which define how many memory units (e.g., bytes) to jump to move one element along that dimension. This ensures correct memory layout interpretation.
    # - The ND coordinates (offsets) of the starting block within the overall tensor.
    # - The shape of the block (tile) that will be loaded or stored in a single operation.
    # - The order of the dimensions in memory from major to minor (e.g., (1, 0) means the second dimension is major, and the first is minor).
    #   This order (axes = np.argsort(strides)) is particularly useful for memory access optimizations, especially on NVIDIA H100 GPUs.

    # Create a block pointer for the input 'x' tensor.
    x_block_ptr = tl.make_block_ptr(
        x_ptr, # Base pointer to the 'x' tensor.
        shape=(ROWS, D,), # Overall shape of the 'x' tensor (total rows, total columns).
        strides=(x_stride_row, x_stride_dim), # Strides for rows and columns of 'x'.
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), # Starting offset for this tile: (current row tile index * tile size, start from column 0).
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), # The shape of the block to be loaded/processed by this instance.
        order=(1, 0), # Memory order: column-major within the block (dimension 1 is major, dimension 0 is minor).
    )

    # Create a block pointer for the 'weight' tensor.
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, # Base pointer to the 'weight' tensor.
        shape=(D,), # Overall shape of the 'weight' tensor (total dimensions).
        strides=(weight_stride_dim,), # Stride for the 1D 'weight' tensor.
        offsets=(0,), # Starting offset for the weights (always from the beginning for each row tile).
        block_shape=(D_TILE_SIZE,), # The shape of the weight block to be loaded.
        order=(0,), # Memory order: standard (dimension 0 is major).
    )

    # Create a block pointer for the 'output' tensor.
    output_block_ptr = tl.make_block_ptr(
        output_ptr, # Base pointer to the 'output' tensor.
        shape=(ROWS,), # Overall shape of the 'output' tensor (total rows).
        strides=(output_stride_row,), # Stride for the 1D 'output' tensor.
        offsets=(row_tile_idx * ROWS_TILE_SIZE,), # Starting offset for this tile in the output vector.
        block_shape=(ROWS_TILE_SIZE,), # The shape of the output block to be stored.
        order=(0,), # Memory order: standard.
    )

    # Initialize a buffer in shared memory (or registers) to accumulate the weighted sum for the current tile of rows.
    # It's a 1D tensor with `ROWS_TILE_SIZE` elements, initialized to zeros, and uses float32 precision.
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    # Loop through the columns of the 'x' tensor, processing them in chunks of D_TILE_SIZE.
    # `tl.cdiv(D, D_TILE_SIZE)` calculates the ceiling division, ensuring all columns are covered even if D is not a multiple of D_TILE_SIZE.
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block (tile) of data from 'x' using the block pointer.
        # `boundary_check=(0, 1)` enables bounds checking for both dimensions (rows and columns) to prevent out-of-bounds access.
        # `padding_option="zero"` pads any out-of-bounds elements with zeros, which is useful for handling partial tiles at the tensor edges.
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # Loads a (ROWS_TILE_SIZE, D_TILE_SIZE) block.
        # Load the corresponding block of weights.
        # `boundary_check=(0,)` enables bounds checking for the single dimension of the weight vector.
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # Loads a (D_TILE_SIZE,) block.

        # Compute the weighted sum for the current tile.
        # `weight[None, :]` reshapes the 1D 'weight' vector to a 2D row vector (1, D_TILE_SIZE) for broadcasting.
        # This allows element-wise multiplication with each row of the 'row' tile (ROWS_TILE_SIZE, D_TILE_SIZE).
        # `tl.sum(..., axis=1)` sums the results along the column dimension (axis 1), producing a (ROWS_TILE_SIZE,) vector.
        # This vector is then added to the accumulated 'output'.
        output += tl.sum(row * weight[None, :], axis=1)

        # Advance the block pointers to the next tile in the column dimension.
        # `x_block_ptr.advance((0, D_TILE_SIZE))` moves the 'x' pointer by D_TILE_SIZE columns (0 rows, D_TILE_SIZE columns).
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE)) # Move by D_TILE_SIZE in the last dimension (columns).
        # `weight_block_ptr.advance((D_TILE_SIZE,))` moves the 'weight' pointer by D_TILE_SIZE elements.
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,)) # Move by D_TILE_SIZE along its single dimension.

    # After processing all column tiles, write the accumulated weighted sum for the current row tile to the output tensor.
    # `boundary_check=(0,)` ensures bounds checking for the output vector, as ROWS_TILE_SIZE might not perfectly divide ROWS.
    tl.store(output_block_ptr, output, boundary_check=(0,))
    
    
def weighted_sum_backward(
    x_ptr, weight_ptr, # Pointers to the original input tensor `x` and weight vector `weight`. These are needed to compute gradients.
    grad_output_ptr, # Pointer to the gradient of the loss with respect to the output of the forward pass (`dL/dy`).
    grad_x_ptr, partial_grad_weight_ptr, # Pointers to the output gradients: `dL/dx` and a partial `dL/dweight`.
    stride_xr, stride_xd, # Strides for the original input `x` (row and dimension strides).
    stride_wd, # Stride for the original `weight` (dimension stride).
    stride_gr, # Stride for `grad_output` (row stride).
    stride_gxr, stride_gxd, # Strides for `grad_x` (row and dimension strides).
    stride_gwb, stride_gwd, # Strides for `partial_grad_weight` (batch/tile stride and dimension stride).
    NUM_ROWS, D, # Total number of rows and dimensions (features) in the problem.
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # Compile-time constants for tile dimensions.
):
    """weighted_sum_backward This code defines the weighted_sum_backward Triton kernel, which computes the gradients of the weighted 
    sum operation with respect to its inputs (x and weight). 

    This kernel would be called by the backward method of the WeightedSumFunc PyTorch autograd function.@triton.jit
    It processes the input tensor `x` and the weight vector `weight`, computes the gradients with respect to them, and stores the 
    results in `grad_x` and `partial_grad_weight`.

    Parameters
    ----------
    x_ptr : 
        _description_
    weight_ptr : 
        _description_
    partial_grad_weight_ptr :
        _description_
    stride_xd : 
        _description_
    stride_gxd : 
        _description_
    stride_gwd : 
        _description_
    D : 
        _description_
    D_TILE_SIZE : tl.constexpr
        _description_
    """
    # Determine the current program instance's ID along the first dimension of the grid.
    # This `row_tile_idx` identifies which block of rows this instance is responsible for.
    row_tile_idx = tl.program_id(0)
    # Get the total number of program instances launched along the first dimension.
    # This is used to define the shape of `partial_grad_weight_ptr`.
    n_row_tiles = tl.num_programs(0)

    # --- Input Block Pointers ---

    # Create a block pointer for `grad_output` (dL/dy).
    # `grad_output` is a 1D tensor, similar to the forward pass output.
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr, # Base pointer to the gradient of the output.
        shape=(NUM_ROWS,), # Overall shape of `grad_output` (total number of rows).
        strides=(stride_gr,), # Stride for `grad_output`.
        offsets=(row_tile_idx * ROWS_TILE_SIZE,), # Offset to the current tile's gradient.
        block_shape=(ROWS_TILE_SIZE,), # Shape of the block to load (1D).
        order=(0,), # Memory order for 1D.
    )

    # Create a block pointer for the original input `x`.
    # This is needed to compute `dL/dweight`.
    x_block_ptr = tl.make_block_ptr(
        x_ptr, # Base pointer to the original input `x`.
        shape=(NUM_ROWS, D,), # Overall shape of `x`.
        strides=(stride_xr, stride_xd), # Strides of `x`.
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), # Offset to the current tile of `x`.
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), # Shape of the block to load.
        order=(1, 0), # Memory order (column-major within the block).
    )

    # Create a block pointer for the original `weight` vector.
    # This is needed to compute `dL/dx`.
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, # Base pointer to the original `weight`.
        shape=(D,), # Overall shape of `weight`.
        strides=(stride_wd,), # Stride of `weight`.
        offsets=(0,), # Offset for `weight` (always from the beginning).
        block_shape=(D_TILE_SIZE,), # Shape of the block to load.
        order=(0,), # Memory order for 1D.
    )

    # --- Output Gradient Block Pointers ---

    # Create a block pointer for `grad_x` (dL/dx).
    # `grad_x` will have the same shape as `x`.
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr, # Base pointer for the gradient of `x`.
        shape=(NUM_ROWS, D,), # Overall shape of `grad_x`.
        strides=(stride_gxr, stride_gxd), # Strides of `grad_x`.
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), # Offset to the current tile of `grad_x`.
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), # Shape of the block to store.
        order=(1, 0), # Memory order (column-major within the block).
    )

    # Create a block pointer for `partial_grad_weight` (dL/dweight).
    # This is a temporary buffer where each program instance will write its partial gradient for `weight`.
    # `shape=(n_row_tiles, D,)`: The first dimension is `n_row_tiles` because each program instance
    # will contribute a 1D gradient for `weight`. These partial gradients will later be summed.
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr, # Base pointer for the partial gradient of `weight`.
        shape=(n_row_tiles, D,), # Overall shape of the partial gradient buffer.
        strides=(stride_gwb, stride_gwd), # Strides for `partial_grad_weight` (batch stride, dimension stride).
        offsets=(row_tile_idx, 0), # Offset: `row_tile_idx` selects the row in this buffer, 0 for columns.
        block_shape=(1, D_TILE_SIZE), # Shape of the block to store (one row of D_TILE_SIZE elements).
        order=(1, 0), # Memory order (column-major within the block).
    )

    # Loop through the columns of the tensors, processing them in chunks of D_TILE_SIZE.
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the gradient of the output for the current tile of rows.
        # `boundary_check=(0,)` handles cases where ROWS_TILE_SIZE doesn't divide NUM_ROWS evenly.
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")
        # `grad_output` will be a (ROWS_TILE_SIZE,) vector.

        # --- Compute Gradient with respect to x (dL/dx) ---
        # The chain rule for `dL/dx` involves `dL/dy * d(y)/dx`.
        # Since `y = x * weight` (element-wise before sum), `d(y)/dx = weight`.
        # So, `dL/dx = dL/dy * weight`. This is an outer product.
        # Load the original `weight` for the current column tile.
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        # `grad_output[:, None]` reshapes (ROWS_TILE_SIZE,) to (ROWS_TILE_SIZE, 1).
        # `weight[None, :]` reshapes (D_TILE_SIZE,) to (1, D_TILE_SIZE).
        # Their multiplication performs an outer product, resulting in a (ROWS_TILE_SIZE, D_TILE_SIZE) matrix.
        grad_x_row = grad_output[:, None] * weight[None, :]
        # Store the computed `grad_x_row` into the `grad_x` tensor.
        # `boundary_check=(0, 1)` handles potential out-of-bounds writes for partial tiles.
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # --- Compute Partial Gradient with respect to weight (dL/dweight) ---
        # The chain rule for `dL/dweight` involves `dL/dy * d(y)/dweight`.
        # Since `y = sum(x * weight)`, `d(y)/dweight = x`.
        # So, `dL/dweight = sum(dL/dy * x, axis=0)`.
        # Load the original input `x` for the current tile.
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        # Element-wise multiply `row` with `grad_output` (broadcasted).
        # `grad_output[:, None]` reshapes (ROWS_TILE_SIZE,) to (ROWS_TILE_SIZE, 1) for broadcasting.
        # `tl.sum(..., axis=0, keep_dims=True)` sums along the row dimension (axis 0),
        # producing a (1, D_TILE_SIZE) vector, which is the partial gradient for `weight` from this tile.
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        # Store this partial gradient into the `partial_grad_weight` buffer.
        # `boundary_check=(1,)` ensures that we don't write out of bounds for the D dimension.
        # The first dimension (batch of tiles) is always in bounds due to `row_tile_idx`.
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)) # Never out of bounds for dim 0

        # --- Advance Block Pointers ---
        # Move all block pointers to the next tile along the `D` (column) dimension.
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE)) # Advance `x` by D_TILE_SIZE columns.
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,)) # Advance `weight` by D_TILE_SIZE elements.
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE)) # Advance partial_grad_weight by D_TILE_SIZE columns.
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE)) # Advance `grad_x` by D_TILE_SIZE columns.



class WeightedSumFunc(torch.autograd.Function):
    """WeightedSumFunc a class that computes the weighted sum of a matrix with a vector.
    
    Written on top of PyTorch's autograd system.

    
    """
    @staticmethod
    def forward(ctx, x, weight):
        """forward method computes the weighted sum of a matrix with a vector.

        This method is called during the forward pass of the autograd system.
        PyTorch’s autograd Functions store state in a special context object (passed as the first argument)
        rather than the Function object, that is why we use `ctx` here and a static method.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionContext
            A context object that can be used to save information for the backward pass.
        x : torch.Tensor
            The input tensor (matrix) to be weighted.
        weight : torch.Tensor
            The weight vector to be applied to the input tensor.
        """
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        # `ctx` is a context object that can be used to save objects for the backward pass.
        # `x` is the input tensor (matrix).
        # `weight` is the weight vector.

        # Determine the last dimension (D) which is the feature/embedding dimension,
        # and the shape of the output tensor (output_dims), which will be the input shape without the last dimension.
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D
        # Our Triton kernel expects a 2D input (ROWS, D).
        # If `x` has more than two dimensions (e.g., a batch of matrices),
        # we flatten all leading dimensions into a single 'rows' dimension.
        input_shape = x.shape # Store the original shape to reshape the output back later.
        x = rearrange(x, "... d -> (...) d") # Uses 'einops.rearrange' to flatten leading dimensions.

        # Save input tensors for the backward pass.
        # `ctx.save_for_backward` is crucial for autograd. It saves the tensors `x` and `weight`
        # so they can be accessed when the `backward` method of this `Function` is called.
        ctx.save_for_backward(x, weight)

        # Assertions for input validation. These ensure the inputs meet the kernel's expectations.
        # 1. Checks if `weight` is a 1D tensor and its size matches `D`.
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        # 2. Checks if both `x` and `weight` are on a CUDA device. Our Triton kernel runs on GPU.
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        # 3. Checks if `x` is contiguous in memory. Our Triton kernel's pointer arithmetic
        #    assumes a contiguous memory layout for efficient access.
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        # Define Triton kernel-specific tile sizes.
        # These are crucial for performance and are passed as `tl.constexpr` to the kernel.
        # `D_TILE_SIZE`: This is the tile size for the feature/embedding dimension (D).
        # `triton.next_power_of_2(D) // 16`: This calculates a power-of-2 tile size for D,
        #    then divides by 16. The comment suggests aiming for roughly 16 loops through D.
        #    This ensures efficient memory access and computation within the kernel.
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16 # Roughly 16 loops through the embedding dimension
        # `ROWS_TILE_SIZE`: This is the tile size for the number of rows.
        #    Each thread block will process 16 rows at a time.
        ctx.ROWS_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time
        # Store the original input shape in `ctx` so it can be restored in the return value.
        ctx.input_shape = input_shape

        # Initialize an empty result tensor `y` on the same device as `x`.
        # `torch.empty` allocates memory but does not initialize it to zeros.
        # The kernel will write the computed values into this tensor.
        y = torch.empty(output_dims, device=x.device)

        # Launch our Triton kernel.
        # `n_rows` is the total number of rows after flattening `x` (i.e., x.shape[0]).
        n_rows = y.numel() # `y.numel()` gives the total number of elements in `y`, which is `ROWS`.

        # Kernel Launch Syntax:
        # `weighted_sum_fwd[...]` is how Triton kernels are launched.
        # `(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)`: This defines the grid dimensions for the kernel launch.
        #    `cdiv` (ceiling division) calculates the number of Triton program instances (thread blocks) needed.
        #    Each instance handles `ctx.ROWS_TILE_SIZE` rows, so we need `n_rows / ctx.ROWS_TILE_SIZE` instances.
        #    The comma `(...,)` makes it a tuple, indicating a 1D grid.
        weighted_sum_fwd[(tl.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, # Input tensors passed to the kernel.
            y, # Output tensor where results will be written.
            x.stride(0), x.stride(1), # Strides of `x` (row stride, column stride).
            weight.stride(0), # Stride of `weight`.
            y.stride(0), # Stride of `y`.
            ROWS=n_rows, D=D, # Total dimensions of the problem.
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE, # Compile-time constants (tile sizes).
        )

        # Reshape the output `y` back to its original leading dimensions.
        # `input_shape[:-1]` removes the last dimension (D) from the original input shape,
        # which is the desired shape for the output `y`.
        return y.view(input_shape[:-1])