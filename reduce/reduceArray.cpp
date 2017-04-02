#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include "reduce.h"

int main(int argc, char* argv[])
{
  // generate randomness
  srand(time(NULL));

  float *h_x;
  long long int N=atoi(argv[1]);//128;

  // allocate memory on host
  h_x = (float*)malloc(N*sizeof(float));

  float sum=0;
  // list initialization & print
  for (long long int i = 0; i < N; i++) {
    h_x[i] = 1;//rand()/100000000;
    sum+=h_x[i];
    //printf("%f\n", h_x[i]);
  }

  // max/min test case
  // h_x[N-2]=10;


  // reduce function
  float rsp = cudaReduce(h_x, N, "sum");
  printf("%f\n", rsp);

  free(h_x);
}
