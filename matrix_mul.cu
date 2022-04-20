#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <vector>

#define VEC_DIM 10000
#define MAT_DIM1 10000
#define MAT_DIM2 10000
#define NUM_THREADS 100
#define NUM_BLOCKS 5

__global__ void custom_matmul(unsigned int *vec, unsigned int *mat,
                              unsigned int *result_ptr) {
  unsigned int result;
  int x = threadIdx.x;

  for (int j = 0; j < MAT_DIM1; j++) {
    result = 0;
    for (int i = 0; i < VEC_DIM; i += NUM_THREADS) {
      result += vec[i + x] * mat[j * MAT_DIM2 + i + x];
    }
    result_ptr[j] = result;
  }
}

int main() {

  std::cout << "start!\n";
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
  }

  curandGenerator_t generator;
  curandRngType_t rng_type = curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT;

  void *mem_ptr = 0; // allocated memory
  cudaMalloc(&mem_ptr, sizeof(int) * VEC_DIM);

  curandCreateGenerator(&generator, rng_type);

  unsigned int *vec_ptr = reinterpret_cast<unsigned int *>(mem_ptr);
  curandGenerate(generator, vec_ptr, VEC_DIM);

  void *mem_ptr2 = 0; // allocated memory
  cudaMalloc(&mem_ptr2, sizeof(int) * MAT_DIM1 * MAT_DIM2);

  unsigned int *mat_ptr = reinterpret_cast<unsigned int *>(mem_ptr2);
  curandGenerate(generator, mat_ptr, MAT_DIM1 * MAT_DIM2);

  unsigned int *result_ptr = 0;
  cudaMalloc(&result_ptr, sizeof(int) * VEC_DIM);

  custom_matmul<<<1, NUM_THREADS>>>(vec_ptr, mat_ptr, result_ptr);
  cudaDeviceSynchronize();

  // test code
  unsigned int *host_ptr =
      reinterpret_cast<unsigned int *>(malloc(sizeof(int) * VEC_DIM));
  cudaMemcpy(host_ptr, result_ptr, sizeof(int) * 200, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++) {
    std::cout << host_ptr[i] << std::endl;
  }
  free(host_ptr);

  curandDestroyGenerator(generator);
  cudaFree(vec_ptr);
  cudaFree(mat_ptr);
  cudaFree(result_ptr);

  std::cout << "finish!";
}