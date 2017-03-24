#include "map.h"

__global__
void stencil(float *x, float *y, int xWidth, int xHeight, float **filter, int filterSize)
{
  int posX = blockIdx.x*blockDim.x + threadIdx.x;
  int posY = blockIdx.y*blockDim.y + threadIdx.y;
  y[posY*xWidth+posX] = x[posY*xWidth+posX];
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
    float *d_x, *d_y, *result, *xTransformed;
    cudaMalloc(&d_x, xWidth*xHeight*sizeof(float));
    cudaMalloc(&d_y, xWidth*xHeight*sizeof(float));
    result = (float*)malloc(xWidth*xHeight*sizeof(float));
    xTransformed = (float*)malloc(xWidth*xHeight*sizeof(float));
    transform2Dto1D(x, xTransformed, xWidth, xHeight);
    // for(int i=0; i<xWidth*xHeight; i++){
    //   printf("%f \n", xTransformed[i]);
    // }
    // allocate memory on device
    //cudaMalloc(&d_x, xWidth*sizeof(float*));
    //for(int i=0; i<xWidth; i++){
    //  cudaMalloc(&d_x[i], xHeight*sizeof(float));
    //}
    // cudaMalloc(&d_y, xWidth*sizeof(float*));
    // for(int i=0; i<xWidth; i++){
    //   cudaMalloc((void**)d_y[i], xHeight*sizeof(float));
    // }
    // copy arrays to device
    cudaMemcpy(d_x, xTransformed, xWidth*xHeight*sizeof(float), cudaMemcpyHostToDevice);
    // for(int i=0; i<xWidth*xHeight; i++){
    //   printf("%f \n", d_x[i]);
    // }
    //cudaMemcpy(d_y, xTransformed, xWidth*xHeight*sizeof(float), cudaMemcpyHostToDevice);

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
    //dim3 threadsPerBlock(2,2);
    //dim3 numOfBlocks(xWidth/2, xHeight/2);

  	stencil<<<numOfBlocks, threadsPerBlock>>>(d_x, d_y, xWidth, xHeight, filter, filterSize);

    // copy array from device to host
    cudaMemcpy(result, d_y, xWidth*xHeight*sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i=0; i<xWidth*xHeight; i++){
    //   printf("%f \n", result[i]);
    // }
    transform1Dto2D(result, x, xWidth, xHeight);
      // show results
      // for (int i = 0; i < xWidth; i++) {
      //   for (int k = 0; k < xHeight; k++) {
      //     printf("%f ", x[i][k]);
      //   }
      //   printf("\n");
      // }
    // free the alocated space
    // for(int i=0; i<xWidth; i++)
    //   cudaFree(d_x[i]);
    cudaFree(d_x);
    // for(int i=0; i<xWidth; i++)
    //   cudaFree(d_y[i]);
    cudaFree(d_y);
}