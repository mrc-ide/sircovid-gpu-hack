#include "add.h"

__global__
void vector_add(const double *a, const double *b, int n, double *value) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    value[i] = a[i] + b[i];
  }
}

void add_gpu(const double *a, const double *b, const int *n, double *value) {
  // Device Memory
  double *d_a, *d_b, *d_value;

  // Define the execution configuration
  dim3 blockSize(256, 1, 1);
  dim3 gridSize(1, 1, 1);
  gridSize.x = (*n + blockSize.x - 1) / blockSize.x;

  // allocate output array
  cudaMalloc((void**)&d_a, *n * sizeof(double));
  cudaMalloc((void**)&d_b, *n * sizeof(double));
  cudaMalloc((void**)&d_value, *n * sizeof(double));

  // copy data to device
  cudaMemcpy(d_a, a, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, *n * sizeof(double), cudaMemcpyHostToDevice);

  // GPU vector add
  vector_add<<<gridSize,blockSize>>>(d_a, d_b, d_value, *n);

  // Copy output
  cudaMemcpy(value, d_value, *n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_value);
}
