#include <cmath>
#include <iostream>
#include <algorithm>

///////////////////////////////////////////////////////////
// Device functions
///////////////////////////////////////////////////////////

__global__
void reduceMin(float *x, float *y, int *n);

__global__
void reduceMax(float *x, float *y, int *n);

__global__
void reduceSum(float *x, float *y);

__global__
void reduceSumSharedAtom(float *x, float *y);

__global__
void reduceSumShared(float *x, float *y);


///////////////////////////////////////////////////////////
// Host functions
///////////////////////////////////////////////////////////

float cudaReduce(float *x, long long int n, std::string functionName);

void reduceSumSharedWrapper(long long int n, int division, float *d_x);
