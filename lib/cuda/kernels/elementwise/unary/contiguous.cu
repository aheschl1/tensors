#include "../../../include/scalar.h"

template <typename T>
__global__ void negate_contiguous_kernel(
    T* __restrict__ data,
    size_t start,
    size_t len
) {
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {
        
        size_t idx = start + i;
        data[idx] = -data[idx];
    }
}

template <typename T>
void launch_negate_contiguous_op(
    T* data,
    size_t start,
    size_t len,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    negate_contiguous_kernel<T><<<grid, block_size>>>(data, start, len);
}

#define DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_negate_contiguous_##SUFFIX( \
        TYPE* data, size_t start, size_t len, unsigned int block_size \
    ) { \
        launch_negate_contiguous_op<TYPE>(data, start, len, block_size); \
    }

DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(double, f64)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int8_t,  i8)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int16_t, i16)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int32_t, i32)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int64_t, i64)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(__int128_t, i128)
