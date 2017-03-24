#include "map.h"

__global__
void mapLog(float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = log(x[i]);
}

__global__
void mapSquare(float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = x[i]*x[i];
}

__global__
void mapSqrt(float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = sqrt(x[i]);
}

__global__
void mapExp(float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = exp(x[i]);
}

__global__
void mapMul(float *x, float *y, const int multiplier)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = x[i] * multiplier;
}

void cudaMap(float *x, int n, std::string functionName, float multiplier){
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