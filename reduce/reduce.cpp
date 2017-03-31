#include "reduce.h"

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
void reduceSum(float *x, float *y) // use only if array length is smaller than 2^10
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  for(int i=blockDim.x/2; i>0; i>>=1){
    if(id<i){
      x[id] += x[id+i];
    }
    __syncthreads();
  }
  if(id==0)
    *y = x[0];
}

__global__
void reduceSumSharedAtom(float *x, float *y) // use only if array length is smaller than 2^25
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  extern __shared__ float s_x[];
  s_x[tid] = x[id];
  __syncthreads();
  for(int i=blockDim.x/2; i>0; i>>=1){
    if(tid<i){
      s_x[tid] += s_x[tid+i];
    }
    __syncthreads();
  }
  if(tid==0)
    atomicAdd(y, s_x[0]);
}

__global__
void reduceSumShared(float *x, float *y) // use only if array length is smaller than 2^25
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  int id = gridDim.x*blockDim.x*idy+idx;

  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tid = blockDim.x*tidy+tidx;
  extern __shared__ float s_x[];
  s_x[tid] = x[id];
  __syncthreads();
  for(int i=blockDim.x*blockDim.y/2; i>0; i>>=1){
    if(tid<i){
      s_x[tid] += s_x[tid+i];
    }
    __syncthreads();
  }
  if(tid==0)
    y[gridDim.x*blockIdx.y+blockIdx.x] = s_x[0];
}


///////////////////////////////////////////////////////////
// Host functions
///////////////////////////////////////////////////////////

float cudaReduce(float *x, long long int n, std::string functionName){
  float *d_x;
  float y;

  // find good division - std::min didnt accepted long long
  int division = (n>1024)?1024:n;
    
  // allocate memory on device
  cudaMalloc(&d_x, n*sizeof(float));
  // copy arrays to devie
  cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);

  // choose funtion
  if(functionName == "min"){
  	//reduceMin<<<n/(division*2), division>>>(d_x, d_y, d_n);
  }
  else if(functionName == "max"){
  	//reduceMax<<<n/(division*2), division>>>(d_x, d_y, d_n);
  }
  else if(functionName == "sum"){
  	reduceSum<<<n/(division), division, division*sizeof(float)>>>(d_x, d_x); // use only if array length is smaller than 2^10
    //reduceSumSharedAtom<<<n/division, division, division*sizeof(float)>>>(d_x, d_y);
    //reduceSumSharedWrapper(n, division, d_x);
  }

  // copy array from device to host
  cudaMemcpy(&y, d_x, sizeof(float), cudaMemcpyDeviceToHost);

  // free the alocated space
  cudaFree(d_x);

  return y;
}

void reduceSumSharedWrapper(long long int n, int division, float *d_x){
  while(n>0){
    int dzielnik = n/division;
    dim3 lol (n/(division*dzielnik), dzielnik, 1);
    reduceSumShared<<<lol, division, division*sizeof(float)>>>(d_x, d_x);
    n>>=10;
    division = (n>1024)?1024:n;
  }
}
