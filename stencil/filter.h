#include <cmath>
#include <stdio.h>

__global__
void stencil(const float * const x, float * const y, const int xWidth, const int xHeight, const float * const filter, const int filterSize, const float filterSum);

void cudaSquareFilter(float ** const x, const int xWidth, const int xHeight, float ** const filter, const int filterSize);

template<typename A>
void transform2Dto1D(A** const arr, A* target, const int width, const int height);

template<typename A>
void transform1Dto2D(A* arr, A** target, const int width, const int height);
