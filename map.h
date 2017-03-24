#include <cmath>
#include <iostream>

__global__
void mapLog(float *x, float *y);

__global__
void mapSquare(float *x, float *y);

__global__
void mapExp(float *x, float *y);

__global__
void mapSqrt(float *x, float *y);

__global__
void mapMul(float *x, float *y, const int multiplier);

void cudaMap(float *x, int n, std::string functionName, float multiplier=1.0);
