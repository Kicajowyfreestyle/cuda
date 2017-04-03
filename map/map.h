#include <cmath>
#include <iostream>

__global__
void mapLog(const float * const x, float * const y);

__global__
void mapSquare(const float * const x, float * const y);

__global__
void mapExp(const float * const x, float * const y);

__global__
void mapSqrt(const float * const x, float * const y);

__global__
void mapMul(const float * const x, float * const y, const float multiplier);

void cudaMap(float *x, const int n, const std::string functionName, const float multiplier=1.0);
