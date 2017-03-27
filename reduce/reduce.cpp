#include "reduce.h"

// TODO: 
//  - shared memory in functions
//  - function to fill data to next power of two with neutral element
//  - adjust other GPU programs to get argument N from comandline parameter
//  - add some time measurement
//  - better division for reducing big arrays


///////////////////////////////////////////////////////////
// Device functions
///////////////////////////////////////////////////////////

__global__
void reduceMin(float *x, float *y, int *n)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  int d_n = *n;
  for(int i=d_n/2; i>0; i>>=1){
    if(id<i){
      x[id] = (x[id]<x[id+i])?x[id]:x[id+i];
    }
    __syncthreads();
  }
  if(id==0)
    *y = x[0];
}

__global__
void reduceMax(float *x, float *y, int *n)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  int d_n = *n;
  for(int i=d_n/2; i>0; i>>=1){
    if(id<i){
      x[id] = (x[id]>x[id+i])?x[id]:x[id+i];
    }
    __syncthreads();
  }
  if(id==0)
    *y = x[0];
}

__global__
void reduceSum(float *x, float *y, int *n)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  int d_n = *n;
  for(int i=d_n/2; i>0; i>>=1){
    if(id<i){
      x[id] += x[id+i];
    }
    __syncthreads();
  }
  if(id==0)
    *y = x[0];
}


///////////////////////////////////////////////////////////
// Host functions
///////////////////////////////////////////////////////////

float cudaReduce(float *x, int n, std::string functionName){
    float *d_x, *d_y, *h_y;
    int *d_n;
    h_y = (float*)malloc(sizeof(float));
    // allocate memory on device
    cudaMalloc(&d_x, n*sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMalloc(&d_n, sizeof(int));
    // copy arrays to devie
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_y, x[0], sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

    // find good division
    int division = std::min(n/2, 1024);

    // choose funtion
  	if(functionName == "min"){
  		reduceMin<<<n/(division*2), division>>>(d_x, d_y, d_n);
  	}
  	else if(functionName == "max"){
  		reduceMax<<<n/(division*2), division>>>(d_x, d_y, d_n);
  	}
  	else if(functionName == "sum"){
  		reduceSum<<<n/(division*2), division>>>(d_x, d_y, d_n);
  	}

    // copy array from device to host
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    float resp = *h_y;
    // free the alocate space
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_y);
    cudaFree(d_n);
    return resp;
}
