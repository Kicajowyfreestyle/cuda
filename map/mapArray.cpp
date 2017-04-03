#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include "map.h"

int main(int argc, char* argv[])
{
  // generate randomness
  srand(time(NULL));

  float *h_x;
  int N=atoi(argv[1]);
  std::string op = argv[2];
  float multiplier = (argc>3)?atof(argv[3]):1.0;

  // allocate memory on host
  h_x = (float*)malloc(N*sizeof(float));

  // list initialization & print
  for (int i = 0; i < N; i++) {
    h_x[i] = rand()/100000;
    printf("%f\n", h_x[i]);
  }

  // map function
  cudaMap(h_x, N, op, multiplier);

  // show results
  for (int i = 0; i < N; i++) {
    printf("%f\n", h_x[i]);
  }

  free(h_x);
}