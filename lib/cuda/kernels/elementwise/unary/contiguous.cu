#include "../../../include/scalar.h"
#include <stdio.h>


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
        // data[idx] = -data[idx];
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


template <typename T>
__global__ void relu_contiguous_kernel(
    T* __restrict__ data,
    size_t start,
    size_t len
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {
        
        size_t idx = start + i;
        if (data[idx] < (T) 0) {
            data[idx] = 0;
        }
    }
}

template <typename T>
void launch_relu_contiguous_op(
    T* data,
    size_t start,
    size_t len,
    unsigned int block_size
) {

    block_size = ALIGN_BLOCK_SIZE(block_size);
    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    relu_contiguous_kernel<T><<<grid, block_size>>>(data, start, len);
}

#define DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_negate_contiguous_##SUFFIX( \
        TYPE* data, size_t start, size_t len, unsigned int block_size \
    ) { \
        launch_negate_contiguous_op<TYPE>(data, start, len, block_size); \
    }

#define DECLARE_RELU_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_relu_contiguous_##SUFFIX( \
        TYPE* data, size_t start, size_t len, unsigned int block_size \
    ) { \
        launch_relu_contiguous_op<TYPE>(data, start, len, block_size); \
    }

DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(double, f64)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int8_t,  i8)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int16_t, i16)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int32_t, i32)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int64_t, i64)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(__int128_t, i128)

DECLARE_RELU_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(double, f64)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(int8_t,  i8)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(int16_t, i16)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(int32_t, i32)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(int64_t, i64)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(__int128_t, i128)



template <typename T>
__global__ void sigmoid_contiguous_kernel(
    T* __restrict__ data,
    size_t start,
    size_t len
) {
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {
        
        size_t idx = start + i;
        data[idx] = ((T) 1) / (((T) 1) + exp(-data[idx]));
    }
}

template <typename T>
void launch_sigmoid_contiguous_op(
    T* data,
    size_t start,
    size_t len,
    unsigned int block_size
) {

    block_size = ALIGN_BLOCK_SIZE(block_size);
    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    sigmoid_contiguous_kernel<T><<<grid, block_size>>>(data, start, len);
}

#define DECLARE_SIGMOID_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_sigmoid_contiguous_##SUFFIX( \
        TYPE* data, size_t start, size_t len, unsigned int block_size \
    ) { \
        launch_sigmoid_contiguous_op<TYPE>(data, start, len, block_size); \
    }


DECLARE_SIGMOID_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_SIGMOID_CONTIGUOUS_LAUNCHER(double, f64)

template <typename T>
__global__ void tanh_contiguous_kernel(
    T* __restrict__ data,
    size_t start,
    size_t len
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {
        
        size_t idx = start + i;

        T a = exp(data[idx]);
        T b = exp(-data[idx]);

        data[idx] = (a - b) / (a + b);
    }
}

template <typename T>
void launch_tanh_contiguous_op(
    T* data,
    size_t start,
    size_t len,
    unsigned int block_size
) {

    block_size = ALIGN_BLOCK_SIZE(block_size);
    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    tanh_contiguous_kernel<T><<<grid, block_size>>>(data, start, len);
}

#define DECLARE_TANH_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_tanh_contiguous_##SUFFIX( \
        TYPE* data, size_t start, size_t len, unsigned int block_size \
    ) { \
        launch_tanh_contiguous_op<TYPE>(data, start, len, block_size); \
    }


DECLARE_TANH_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_TANH_CONTIGUOUS_LAUNCHER(double, f64)

#include <cub/cub.cuh>

template <typename T>
__global__ void test_sum_kernel(
    T* __restrict__ data,
    size_t start,
    size_t len
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {
        
        size_t idx = start + i;

        printf("hello: %f %d %d\n", data[idx], blockIdx.x, threadIdx.x);

    
    }
}

 // CustomMin functor
    struct CustomMin
    {
        template <typename T>
        __device__ __forceinline__
        T operator()(const T &a, const T &b) const {
            return (b < a) ? b : a;
        }
    };


template <typename T>
void launch_test_sum_op(
    T* data,
    size_t start,
    size_t len,
    unsigned int block_size
) {


    printf("Hello\n");
    cudaDeviceSynchronize();
    
    

   
    // Declare, allocate, and initialize device-accessible pointers for
    // input and output
    int          num_items = len;  // e.g., 7
    T          *d_in = &data[start];      // e.g., [8, 6, 7, 5, 3, 0, 9]
    T          *d_out = nullptr;     // e.g., [-]
    CustomMin    min_op;
    T          init = (T) 10000;       // e.g., INT_MAX


    cudaMalloc(&d_out, sizeof(T));

    // Determine temporary device storage requirements
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes,
    d_in, d_out, num_items, min_op, init);

    // printf("Parameters: num_items=%d, d_in=%f, d_out=%d\n", num_items, data[0], 23);
    cudaDeviceSynchronize();

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run reduction
    cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes,
    d_in, d_out, num_items, min_op, init);


    T h0;
    cudaMemcpy(&h0, d_out, sizeof(T), cudaMemcpyDeviceToHost);
    printf("first element = %g\n", (double)h0);  // cast for safe printf


    cudaDeviceSynchronize();



    // block_size = ALIGN_BLOCK_SIZE(block_size);
    // const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    // test_sum_kernel<T><<<grid, block_size>>>(data, start, len);
}

// #define DECLARE_TANH_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
//     extern "C" void launch_tanh_contiguous_##SUFFIX( \
//         TYPE* data, size_t start, size_t len, unsigned int block_size \
//     ) { \
//         launch_tanh_contiguous_op<TYPE>(data, start, len, block_size); \
//     }


extern "C" void launch_test_summy(double* data, size_t start, size_t len, unsigned int block_size) {
    launch_test_sum_op(data, start, len, block_size);
}

