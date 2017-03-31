#include <cmath>
#include <iostream>
#include <algorithm>

///////////////////////////////////////////////////////////
// Device functions
///////////////////////////////////////////////////////////

__global__
void reduceMin(float *x, float *y);

__global__
void reduceMinShared(float *x, float *y);

__global__
void reduceMax(float *x, float *y);

__global__
void reduceMaxShared(float *x, float *y);

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

void reduceSumSharedWrapper(long long int n, int division, float *d_x, float *d_y);

void reduceMinSharedWrapper(long long int n, int division, float *d_x, float *d_y);

void reduceMaxSharedWrapper(long long int n, int division, float *d_x, float *d_y);

///////////////////////////////////////////////////////////
// Tool functions
///////////////////////////////////////////////////////////

template <typename T>
T min(T a, T b);

template <typename T>
T max(T a, T b);
