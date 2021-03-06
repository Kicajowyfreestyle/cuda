#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include "filter.h"

int main(int argc, char* argv[])
{
  // generate randomness
  srand(time(NULL));

  float **h_x, **filter;
  int N          = (argc>1) ? atoi(argv[1]) : 4;
  int filterSize = (argc>2) ? atoi(argv[2]) : 3;

  // allocate memory on host
  h_x = (float**)malloc(N*sizeof(float*));
  filter = (float**)malloc(filterSize*sizeof(float*));

  // array initialization & print
  for (int i = 0; i < N; i++) {
    h_x[i] = (float*)malloc(N*sizeof(float));
    for (int k = 0; k < N; k++) {
      h_x[i][k] = rand()/100000;
      printf("%f ", h_x[i][k]);
    }
    printf("\n");
  }

  // create filter
  for (int i = 0; i < filterSize; i++) {
    filter[i] = (float*)malloc(filterSize*sizeof(float));
    for (int k = 0; k < filterSize; k++) {
      filter[i][k] = (float)(i+k)/10;
      printf("%f ", filter[i][k]);
    }
    printf("\n");
  }

  // filter function
  cudaSquareFilter(h_x, N, N, filter, filterSize);

  // show results
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      printf("%f ", h_x[i][k]);
    }
    printf("\n");
  }


  // free allocated memory
  for (int i = 0; i < N; i++) {
    free(h_x[i]);
  }
  free(h_x);

  for (int i = 0; i < filterSize; i++) {
    free(filter[i]);
  }
  free(filter);
}