#include "reduce.h"

///////////////////////////////////////////////////////////
// Device functions
///////////////////////////////////////////////////////////

__global__
void reduceMin(float *x, float *y) // use only if array length is smaller or equal than 2^10
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  for(int i=blockDim.x/2; i>0; i>>=1){
    if(id<i){
      x[id] = MIN(x[id], x[id+i]);//(x[id]<x[id+i])?x[id]:x[id+i];
    }
    __syncthreads();
  }
  if(id==0)
    *y = x[0];
}

__global__
void reduceMinShared(float *x, float *y) // use only if array length is smaller or equal than 2^25
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
       s_x[tid] = MIN(s_x[tid], s_x[tid+i]);
    }
    __syncthreads();
  }
  if(tid==0){
    int blockZeroElementId = gridDim.x*blockIdx.y+blockIdx.x;
    y[blockZeroElementId] = MIN(s_x[0], y[blockZeroElementId]);
  }
}


__global__
void reduceMax(float *x, float *y) // use only if array length is smaller or equal than 2^10
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  for(int i=blockDim.x/2; i>0; i>>=1){
    if(id<i){
      x[id] = MAX(x[id], x[id+i]);
    }
    __syncthreads();
  }
  if(id==0)
    *y = x[0];
}

__global__
void reduceMaxShared(float *x, float *y) // use only if array length is smaller or equal than 2^25
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
       s_x[tid] = MAX(s_x[tid], s_x[tid+i]);
    }
    __syncthreads();
  }
  if(tid==0){
    int blockZeroElementId = gridDim.x*blockIdx.y+blockIdx.x;
    y[blockZeroElementId] = MAX(s_x[0], y[blockZeroElementId]);
  }
}


__global__
void reduceSum(float *x, float *y) // use only if array length is smaller or equal than 2^10
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
void reduceSumSharedAtom(float *x, float *y) // use only if array length is smaller or equal than 2^25
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
void reduceSumShared(float *x, float *y) // use only if array length is smaller or equal than 2^25
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

float cudaReduce(float *x, long long int numberOfElements, std::string functionName)
{
  float *d_x;
  float y;
  
  // find good division - std::min didnt accepted long long
  int division = MIN(numberOfElements, 1024);

  // choose funtion
  // ommented out lines left for performace test
  if(functionName == "min"){
    fillWithNeutral(x, numberOfElements, std::numeric_limits<float>::max());
    // allocate memory on device
    cudaMalloc(&d_x, numberOfElements*sizeof(float));
    // copy arrays to devie
    cudaMemcpy(d_x, x, numberOfElements*sizeof(x[0]), cudaMemcpyHostToDevice);

  	//reduceMin<<<numberOfElements/division, division>>>(d_x, d_x);
    reduceMinSharedWrapper(numberOfElements, division, d_x, d_x);
  }
  else if(functionName == "max"){
    fillWithNeutral(x, numberOfElements, -std::numeric_limits<float>::max());
    // allocate memory on device
    cudaMalloc(&d_x, numberOfElements*sizeof(float));
    // copy arrays to devie
    cudaMemcpy(d_x, x, numberOfElements*sizeof(x[0]), cudaMemcpyHostToDevice);

    //reduceMax<<<numberOfElements/division, division>>>(d_x, d_x);
    reduceMaxSharedWrapper(numberOfElements, division, d_x, d_x);
  }
  else if(functionName == "sum"){
    fillWithNeutral(x, numberOfElements, 0);
    // allocate memory on device
    cudaMalloc(&d_x, numberOfElements*sizeof(float));
    // copy arrays to devie
    cudaMemcpy(d_x, x, numberOfElements*sizeof(x[0]), cudaMemcpyHostToDevice);

  	//reduceSum<<<numberOfElements/(division), division>>>(d_x, d_x); // use only if array length is smaller or equal than 2^10
    //reduceSumSharedAtom<<<numberOfElements/division, division, division*sizeof(x[0])>>>(d_x, d_y);
    reduceSumSharedWrapper(numberOfElements, division, d_x, d_x);
  }

  // copy array from device to host
  cudaMemcpy(&y, d_x, sizeof(y), cudaMemcpyDeviceToHost);

  // free the alocated space
  cudaFree(d_x);

  return y;
}

// TODO: make a class with wirtual method from that

void reduceSumSharedWrapper(long long int numberOfElements, int division, float *d_x, float *d_y)
{
  while(numberOfElements>0){
    int dzielnik = numberOfElements/division;
    dim3 lol (numberOfElements/(division*dzielnik), dzielnik, 1);
    reduceSumShared<<<lol, division, division*sizeof(d_x[0])>>>(d_x, d_y);
    numberOfElements>>=10;
    division = (numberOfElements>1024)?1024:numberOfElements;
  }
}

void reduceMinSharedWrapper(long long int numberOfElements, int division, float *d_x, float *d_y)
{
  while(numberOfElements>0){
    int dzielnik = numberOfElements/division;
    dim3 lol (numberOfElements/(division*dzielnik), dzielnik, 1);
    reduceMinShared<<<lol, division, division*sizeof(d_x[0])>>>(d_x, d_y);
    numberOfElements>>=10;
    division = (numberOfElements>1024)?1024:numberOfElements;
  }
}

void reduceMaxSharedWrapper(long long int numberOfElements, int division, float *d_x, float *d_y)
{
  while(numberOfElements>0){
    int dzielnik = numberOfElements/division;
    dim3 lol (numberOfElements/(division*dzielnik), dzielnik, 1);
    reduceMaxShared<<<lol, division, division*sizeof(d_x[0])>>>(d_x, d_y);
    numberOfElements>>=10;
    division = (numberOfElements>1024)?1024:numberOfElements;
  }
}

///////////////////////////////////////////////////////////
// Tool functions
///////////////////////////////////////////////////////////

void fillWithNeutral(float *array, long long int& numberOfElements, float neutralElement){
  long long int stop = nextPowerOf2(numberOfElements);
  array = (float*)realloc(array, stop*sizeof(array[0]));
  std::fill(array+numberOfElements, array+stop, neutralElement);
  numberOfElements = stop;
}

long long int nextPowerOf2(long long int n){
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;
  return n;
}
// template <typename T>
// T min(T a, T b){
//   return (a<b)?a:b;
// }

// template <typename T>
// T max(T a, T b){
//   return (a>b)?a:b;
// }
