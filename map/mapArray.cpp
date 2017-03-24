#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include "map.h"

int main(void)
{
  // generate randomness
  srand(time(NULL));

  float *h_x;
  int N=1024;

  // allocate memory on host
  h_x = (float*)malloc(N*sizeof(float));

  // list initialization & print
  for (int i = 0; i < N; i++) {
    h_x[i] = rand()/100000;
    printf("%f\n", h_x[i]);
  }

  // map function
  cudaMap(h_x, N, "log");

  // show results
  for (int i = 0; i < N; i++) {
    printf("%f\n", h_x[i]);
  }

  free(h_x);
}