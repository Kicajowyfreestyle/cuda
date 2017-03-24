#include <cmath>
#include <stdio.h>

__global__
void stencil(float *x, float *y, int xWidth, int xHeight, float **filter, int filterSize);

void cudaSquareFilter(float **x, int xWidth, int xHeight, float **filter, int filterSize);

template<typename A>
void transform2Dto1D(A** arr, A* target, int width, int height);

template<typename A>
void transform1Dto2D(A* arr, A** target, int width, int height);
