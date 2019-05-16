#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/UpSample.cuh>

namespace at {
namespace native {
namespace {

static int lastPow2(unsigned int n) {
  n |= (n >>  1);
  n |= (n >>  2);
  n |= (n >>  4);
  n |= (n >>  8);
  n |= (n >> 16);
  return n - (n >> 1);
}

template <typename scalar_t, typename accscalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void upsample_nearest2d_out_frame(
    const PackedTensorAccessor<scalar_t, 3> idata,
    PackedTensorAccessor<scalar_t, 3> odata) {
  int nc_iter = threadIdx.z + blockIdx.z * blockDim.z;
  int w2 = threadIdx.x + blockIdx.x * blockDim.x;
  int h2 = threadIdx.y + blockIdx.y * blockDim.y;

  const int nc = idata.size(0);
  const int height1 = idata.size(1);
  const int width1 = idata.size(2);
  const int height2 = odata.size(1);
  const int width2 = odata.size(2);

  if ( w2 >= width2 || h2 >= height2 ) {
    return;
  }

  const float height_scale = (float)height1 / (float)height2;
  const float width_scale = (float)width1 / (float)width2;
  int nc_stride = blockDim.z * gridDim.z;

  const int h1 = height1 == height2 ? h2 :
      nearest_neighbor_compute_source_index(height_scale, h2, height1);
  const int w1 = width1 == width2 ? w2 :
      nearest_neighbor_compute_source_index(width_scale, w2, width1);

  // iterating over 
  while (nc_iter < nc) {
    const scalar_t val = idata[nc_iter][h1][w1];
    odata[nc_iter][h2][w2] = val;
    nc_iter += nc_stride;
  }
}

// Backward operation
template <typename scalar_t, typename accscalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void upsample_nearest2d_backward_out_frame(
    const int width_per_block,
    PackedTensorAccessor<scalar_t, 3> idata,
    const PackedTensorAccessor<scalar_t, 3> odata) {
  // shared memory used for reduction;
  // TODO: pad memory to avoid bank conflicts;
  extern __shared__ char smem[];
  
  const int nc = idata.size(0);
  const int height1 = idata.size(1);
  const int width1 = idata.size(2);
  const int height2 = odata.size(1);
  const int width2 = odata.size(2);

  //TODO: think twice! how about sync calls later?
  // if ( nc_index >= nc || w2 >= width2 || h1 >= height2 ) {
  //   return;
  // }

  const float height_scale = (float)height1 / (float)height2;
  const float width_scale = (float)width1 / (float)width2;

  int nc_index = threadIdx.z + blockIdx.z * blockDim.z;
  // offset computed.
  int block_offset_w1 = blockIdx.x * width_per_block;
  int w2 = threadIdx.x + 
      nearest_neighbor_compute_destination_index(width_scale, block_offset_w1, width2);
  int h1 = threadIdx.y + blockIdx.y * blockDim.y;
  int t_id = 
      threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
      + threadIdx.x;

  accscalar_t acc = 0.0;

  // accumulation across column
  const int h2_offset =
      nearest_neighbor_compute_destination_index(height_scale, h1, height2);
  const int h2_boundary =
      nearest_neighbor_compute_destination_index(height_scale, h1+1, height2);
  for (int h2 = h2_offset; h2 < h2_boundary && w2 < width2; h2++) {
    acc += odata[nc_index][h2][w2];
  }

  // write to shared_mem
  accscalar_t* buffer = (accscalar_t*)smem;
  int index =
      threadIdx.x * (blockDim.y * blockDim.z + 1) + threadIdx.y * blockDim.z
      + threadIdx.z;
  buffer[index] = acc;

  __syncthreads();

  // accumulation across row and write to output
  if (t_id <= blockDim.z * blockDim.y * width_per_block) {
    // adjust block layout to accommodate shrinked block x dimension.
    const int layout_x = t_id % width_per_block;
    int tmp_id = t_id / width_per_block;
    const int layout_y = tmp_id % blockDim.y;
    const int layout_z = tmp_id / blockDim.y;

    acc = 0.0;
    // accumulate across row;
    int w1 = layout_x + blockIdx.x * width_per_block;
    nc_index = layout_z + blockIdx.z * blockDim.z;
    h1 = layout_y + blockIdx.y * blockDim.y;
    const int acc_width_length =
        nearest_neighbor_compute_destination_index(width_scale, w1+1, width2);
        - nearest_neighbor_compute_destination_index(width_scale, w1, width2);
    int offset = layout_x * (blockDim.y * blockDim.z + 1) + layout_y * blockDim.z
                 + layout_z;
    int stride = blockDim.y * blockDim.z;
    for (int i = 0; i < acc_width_length && h1 < height1 && nc_index < nc; i++) {
      acc += buffer[offset];
      offset += stride;
    }

    // write output
    idata[nc_index][h1][w1] = acc;
  }
}

static void upsample_nearest2d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(
      "upsample_nearest2d_out_cuda_template", {input_arg, output_arg});

  AT_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  output.resize_({nbatch, channels, output_height, output_width});

  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  AT_ASSERT(
      input_height > 0 && input_width > 0 && output_height > 0 &&
      output_width > 0);

  int nc = input.size(0) * input.size(1);
  Tensor input_projected = input;
  Tensor output_projected = output;
  input_projected.resize_({nc, input_height, input_width});
  output_projected.resize_({nc, output_height, output_width});

  const int max_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

  int block_x = std::min<int>(lastPow2(output_width), max_threads);
  int block_y = std::min<int>(lastPow2(output_height), max_threads/block_x);
  int block_z = std::min<int>(nc, max_threads/block_x/block_y);
  const dim3 block(block_x, block_y, block_z);

  int grid_x = cuda::ATenCeilDiv(output_width, block_x);
  int grid_y = cuda::ATenCeilDiv(output_height, block_y);
  // maybe we should add a loop here;
  int grid_z = cuda::ATenCeilDiv(nc, block_z*4);
  const dim3 grid(grid_x, grid_y, grid_z);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_nearest2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input_projected.packed_accessor<scalar_t, 3>();
        auto odata = output_projected.packed_accessor<scalar_t, 3>();

        upsample_nearest2d_out_frame<scalar_t, accscalar_t>
            <<<grid,
               block,
               0,
               stream>>>(idata, odata);
      });

  input_projected.resize_({nbatch, channels, input_height, input_width});
  output_projected.resize_({nbatch, channels, output_height, output_width});
  AT_CUDA_CHECK(cudaGetLastError());
}

static void upsample_nearest2d_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_nearest2d_backward_out_cuda",
      {grad_output_arg, grad_input_arg});

  AT_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  AT_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];

  upsample_2d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  int nc = nbatch * channels;

  Tensor grad_output = grad_output_.contiguous();
  grad_input.resize_({nc, input_height, input_width});
  grad_output.resize_({nc, output_height, output_width});

  const int max_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

  int block_x = std::min<int>(lastPow2(output_width), max_threads/4);
  // this is intended to be a floorf(block.x / (float)width_scale);
  int width_per_block = block_x * input_width / output_width;
  // we let each thread loop over reduced-column;
  int block_y = std::min<int>(lastPow2(input_height), max_threads/block_x);
  int block_z = std::min<int>(nc, max_threads/block_x/block_y);
  const dim3 block(block_x, block_y, block_z);

  int grid_x = cuda::ATenCeilDiv(output_width, block_x);
  int grid_y = cuda::ATenCeilDiv(input_height, block_y);
  int grid_z = cuda::ATenCeilDiv(nc, block_z);
  const dim3 grid(grid_x, grid_y, grid_z);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "upsample_nearest2d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor<scalar_t, 3>();
        auto odata = grad_output.packed_accessor<scalar_t, 3>();

        // shared memory used for row reduction;
        // padded to avoid bank conflict;
        size_t mem_size = block_x * (1 + block_y * block_z) * sizeof(scalar_t);

        upsample_nearest2d_backward_out_frame<scalar_t, accscalar_t>
            <<<grid,
               block,
               mem_size,
               stream>>>(width_per_block, idata, odata);
      });

  grad_input.resize_({nbatch, channels, input_height, input_width});
  grad_output.resize_({nbatch, channels, output_height, output_width});
  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

Tensor& upsample_nearest2d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  upsample_nearest2d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor upsample_nearest2d_cuda(const Tensor& input, IntArrayRef output_size) {
  Tensor output = at::empty_like(input);
  upsample_nearest2d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor& upsample_nearest2d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  upsample_nearest2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

Tensor upsample_nearest2d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  Tensor grad_input = at::empty_like(grad_output);
  upsample_nearest2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

} // namespace native
} // namespace at
