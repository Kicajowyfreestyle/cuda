#include "filter.h"

__global__
void stencil(float *x, float *y, int xWidth, int xHeight, float *filter, int filterSize, float filterSum)
{
  int posX = blockIdx.x*blockDim.x + threadIdx.x;
  int posY = blockIdx.y*blockDim.y + threadIdx.y;
  //y[posY*xWidth+posX] = x[posY*xWidth+posX];
  float sum = 0.0;
  float subFromSum = 0.0;
  float subFromSumPrev = 0.0;
  int range = filterSize/2;
  for(int i=-range; i<=range; i++){
    for(int k=-range; k<=range; k++){
      if(i+posY<0)
        subFromSum+=filter[k+range];
      else if(i+posY>=xHeight)
        subFromSum+=filter[(filterSize-1)*filterSize+k+range];
      else if(k+posX<0)
        subFromSum+=filter[(i+range)*filterSize];
      else if(k+posX>=xWidth)
        subFromSum+=filter[(i+range)*filterSize+filterSize-1];
      else
         sum+=x[(i+posY)*xWidth+k+posX]*filter[(i+range)*filterSize+k+range];
    }
  }
  y[posY*xWidth+posX] =  sum/(filterSum-subFromSum);
}

template<typename A>
void transform2Dto1D(A** arr, A* target, int width, int height){
  for(int i=0; i<width; i++){
    for(int k=0; k<height; k++){
      target[i*width+k] = arr[i][k];
    }
  }
}

template<typename A>
void transform1Dto2D(A* arr, A** target, int width, int height){
  for(int i=0; i<width; i++){
    for(int k=0; k<height; k++){
      target[i][k] = arr[i*width+k];
    }
  }
}


void cudaSquareFilter(float **x, int xWidth, int xHeight, float **filter, int filterSize){
    float *d_x, *d_y, *d_filter, *result, *xTransformed, *filterTransformed;

    cudaMalloc(&d_x, xWidth*xHeight*sizeof(float));
    cudaMalloc(&d_y, xWidth*xHeight*sizeof(float));
    cudaMalloc(&d_filter, filterSize*filterSize*sizeof(float));

    result = (float*)malloc(xWidth*xHeight*sizeof(float));
    xTransformed = (float*)malloc(xWidth*xHeight*sizeof(float));
    filterTransformed = (float*)malloc(filterSize*filterSize*sizeof(float));

    // transform 2d input into 1d for cuda
    transform2Dto1D(x, xTransformed, xWidth, xHeight);
    transform2Dto1D(filter, filterTransformed, filterSize, filterSize);

    // copy arrays to device
    cudaMemcpy(d_x, xTransformed, xWidth*xHeight*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filterTransformed, filterSize*filterSize*sizeof(float), cudaMemcpyHostToDevice);

    // calculate filter sum
    float filterSum=0.0;
    for (int i = 0; i < filterSize; i++) {
      for (int k = 0; k < filterSize; k++) {
        filterSum+=filter[i][k];
      }
    }

    // find good division
    int divisionWidth = 2;
    for(;divisionWidth<=16 && divisionWidth<xWidth/4; divisionWidth = divisionWidth<<1)
      if(xWidth%divisionWidth) break;
    int divisionHeight = 2;
    for(;divisionHeight<=16 && divisionHeight<xHeight/4; divisionHeight = divisionHeight<<1)
      if(xWidth%divisionHeight) break;

    // dims for gpu
    dim3 threadsPerBlock(divisionWidth, divisionHeight);
    dim3 numOfBlocks(xWidth/threadsPerBlock.x, xHeight/threadsPerBlock.y);

  	stencil<<<numOfBlocks, threadsPerBlock>>>(d_x, d_y, xWidth, xHeight, d_filter, filterSize, filterSum);

    // copy array from device to host
    cudaMemcpy(result, d_y, xWidth*xHeight*sizeof(float), cudaMemcpyDeviceToHost);
    // come back to the 2d array
    transform1Dto2D(result, x, xWidth, xHeight);

    // free the alocated space
    cudaFree(d_x);
    cudaFree(d_y);
    free(result);
    free(xTransformed);
}