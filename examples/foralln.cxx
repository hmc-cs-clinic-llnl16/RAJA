#ifndef RAJA_ENABLE_NESTED
#define RAJA_ENABLE_NESTED
#endif

#include "RAJA/RAJA.hxx"
#include <cstdlib>

typedef RAJA::simd_exec policy_one;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec> > policy_n;
typedef RAJA::omp_reduce reduce_policy;


int main(int argc, char* argv[]) {
  const int n = 150;

  double* a = new double[n*n];
  double* b = new double[n*n];

  RAJA::RangeSegment is(0,n);
  RAJA::RangeSegment js(0,n);


  RAJA::forallN<policy_n>(is, js, [=] (int i, int j) {
    a[i*n+j] = i + 0.1*j;
    b[i*n+j] = abs(i-j);
  });


  RAJA::ReduceSum<reduce_policy, double> sum(0.0);

  RAJA::forallN<policy_n>(is, js, [=] (int i, int j) {
      sum += a[i*n+j]*b[i*n+j];
  });

  printf("Sum is %f\n", double(sum));
}
