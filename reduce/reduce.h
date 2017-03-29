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
void reduceSum(float *x, float *y, int *n);

__global__
void reduceSumSharedAtom(float *x, float *y);

///////////////////////////////////////////////////////////
// Host functions
///////////////////////////////////////////////////////////

float cudaReduce(float *x, int n, std::string functionName);
