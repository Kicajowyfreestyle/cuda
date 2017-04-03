#include "map.h"

__global__
void mapLog(const float * const x, float * const y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = log(x[i]);
}

__global__
void mapSquare(const float * const x, float * const y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = x[i]*x[i];
}

__global__
void mapSqrt(const float * const x, float * const y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = sqrt(x[i]);
}

__global__
void mapExp(const float * const x, float * const y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = exp(x[i]);
}

__global__
void mapMul(const float * const x, float * const y, const float multiplier)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = x[i] * multiplier;
}

void cudaMap(float * x, const int n, const std::string functionName, const float multiplier){
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

    // choose funtion
  	if(functionName == "log"){
  		mapLog<<<n*2/division, division/2>>>(d_x, d_y);
  	}
  	else if(functionName == "sqare"){
  		mapSquare<<<n*2/division, division/2>>>(d_x, d_y);
  	}
  	else if(functionName == "exp"){
  		mapExp<<<n*2/division, division/2>>>(d_x, d_y);
  	}
  	else if(functionName == "sqrt"){
  		mapSqrt<<<n*2/division, division/2>>>(d_x, d_y);
  	}
  	else if(functionName == "mul"){
  		mapMul<<<n*2/division, division/2>>>(d_x, d_y, multiplier);
  	}

    // copy array from device to host
    cudaMemcpy(x, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);
    // free the alocate space
    cudaFree(d_x);
    cudaFree(d_y);
}