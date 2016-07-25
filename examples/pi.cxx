#include <cstdlib>
#include <iostream>
#include "RAJA/RAJA.hxx"
//#include "RAJA/exec-cuda/raja_cuda.hxx"

template<typename LB>
__host__ __device__ void stupid_launcher(int start, int end, LB body){
    for(int i=start;i<end;i++){
        body(i);
    }
}

int main(int argc, char* argv[])
{
  typedef RAJA::cuda_reduce<32> reduce_policy;
  typedef RAJA::cuda_exec<32> execute_policy;

  int numBins = 512 * 512;
  //double piSum = 0.0;
  RAJA::ReduceSum<reduce_policy, double> piSum(0.0);

  RAJA::forall<execute_policy>(0, numBins,  [=] (int i) __host__ __device__ {
    double x = (double(i) + 0.5) / numBins;
    piSum += 4.0 / (1.0 + x * x);
  });

  std::cout << "PI is ~ " << double(piSum) / (numBins) << std::endl;

  return 0;
}
