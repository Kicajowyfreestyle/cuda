#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cmath>

__global__
void mapLog(float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = log(x[i]);
}

void cudaMapLog(float *x, int n){
    float *d_x, *d_y;
    // allocate memory on device
    cudaMalloc(&d_x, n*sizeof(float));
    cudaMalloc(&d_y, n*sizeof(float));
    // copy arrays to devie
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, x, n*sizeof(float), cudaMemcpyHostToDevice);

    // find good division
    int division = 2;
    for(;division<=1024; division = division<<1)
      if(n%division) break;

    mapLog<<<n*2/division, division/2>>>(d_x, d_y);

    cudaMemcpy(x, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}

int main(void)
{
  // generate randomness
  srand(time(NULL));

  float *h_x;
  int N=1024;

  // allocate memory on host
  h_x = (float*)malloc(N*sizeof(float));

  // list initialization & print
  for (int i = 0; i < N; i++) {
    h_x[i] = rand()/100000;
    printf("%f\n", h_x[i]);
  }

  // map function
  cudaMapLog(h_x, N);

  // show results
  for (int i = 0; i < N; i++) {
    printf("%f\n", h_x[i]);
  }

  free(h_x);
}