/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

#include <math.h>
#include <cfloat>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "RAJA/RAJA.hxx"

#include "Compare.hxx"

#define TEST_VEC_LEN 1024 * 1024 * 15

using namespace RAJA;
using namespace std;

///
/// Define thread block size for CUDA exec policy
///
const size_t block_size = 256;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

const int test_repeat = 10;

int tcount = 0;

double dinit_val = 0.1;
int iinit_val = 1;
__managed__ double *dvalue;
__managed__ int *ivalue;


//
// test 1 runs 8 reductions over a range multiple times to check
//        that reduction value can be retrieved and then subsequent
//        reductions can be run with the same reduction objects.
//
template<typename execution_type, typename reduction_type>
void test1()
{  // begin test 1

  double dtinit = 5.0;

  ReduceSum<reduction_type, double> dsum0(0.0);
  ReduceSum<reduction_type, double> dsum1(dtinit * 1.0);
  ReduceSum<reduction_type, double> dsum2(0.0);
  ReduceSum<reduction_type, double> dsum3(dtinit * 3.0);
  ReduceSum<reduction_type, double> dsum4(0.0);
  ReduceSum<reduction_type, double> dsum5(dtinit * 5.0);
  ReduceSum<reduction_type, double> dsum6(0.0);
  ReduceSum<reduction_type, double> dsum7(dtinit * 7.0);

  int loops = 2;
  for (int k = 0; k < loops; k++) {
    s_ntests_run++;

    forall<execution_type>(0, TEST_VEC_LEN, [=] __device__(int i) {
      dsum0 += dvalue[i];
      dsum1 += dvalue[i] * 2.0;
      dsum2 += dvalue[i] * 3.0;
      dsum3 += dvalue[i] * 4.0;
      dsum4 += dvalue[i] * 5.0;
      dsum5 += dvalue[i] * 6.0;
      dsum6 += dvalue[i] * 7.0;
      dsum7 += dvalue[i] * 8.0;
    });

    double base_chk_val = dinit_val * double(TEST_VEC_LEN) * (k + 1);

    if (!equal(double(dsum0), base_chk_val)
        || !equal(double(dsum1), 2 * base_chk_val + (dtinit * 1.0))
        || !equal(double(dsum2), 3 * base_chk_val)
        || !equal(double(dsum3), 4 * base_chk_val + (dtinit * 3.0))
        || !equal(double(dsum4), 5 * base_chk_val)
        || !equal(double(dsum5), 6 * base_chk_val + (dtinit * 5.0))
        || !equal(double(dsum6), 7 * base_chk_val)
        || !equal(double(dsum7), 8 * base_chk_val + (dtinit * 7.0))) {
      cout << "\n TEST 1 <" <<  typeid(execution_type).name() << ", " << typeid(reduction_type).name() << "> FAILURE: tcount, k = " << tcount << " , " << k
           << endl;
      cout << setprecision(20) << "\tdsum0 = " << static_cast<double>(dsum0)
           << " (" << base_chk_val << ") " << endl;
      cout << setprecision(20) << "\tdsum1 = " << static_cast<double>(dsum1)
           << " (" << 2 * base_chk_val + (dtinit * 1.0) << ") " << endl;
      cout << setprecision(20) << "\tdsum2 = " << static_cast<double>(dsum2)
           << " (" << 3 * base_chk_val << ") " << endl;
      cout << setprecision(20) << "\tdsum3 = " << static_cast<double>(dsum3)
           << " (" << 4 * base_chk_val + (dtinit * 3.0) << ") " << endl;
      cout << setprecision(20) << "\tdsum4 = " << static_cast<double>(dsum4)
           << " (" << 5 * base_chk_val << ") " << endl;
      cout << setprecision(20) << "\tdsum5 = " << static_cast<double>(dsum5)
           << " (" << 6 * base_chk_val + (dtinit * 5.0) << ") " << endl;
      cout << setprecision(20) << "\tdsum6 = " << static_cast<double>(dsum6)
           << " (" << 7 * base_chk_val << ") " << endl;
      cout << setprecision(20) << "\tdsum7 = " << static_cast<double>(dsum7)
           << " (" << 8 * base_chk_val + (dtinit * 7.0) << ") " << endl;

    } else {
      s_ntests_passed++;
    }
  }

}  // end test 1


//
// test 2 runs 4 reductions (2 int, 2 double) over complete array
//        using an indexset with two range segments to check
//        reduction object state is maintained properly across
//        kernel invocations.
//        Also exercises the get function call
template<typename execution_type, typename reduction_type>
void test2()
{  // begin test 2

  s_ntests_run++;

  RangeSegment seg0(0, TEST_VEC_LEN / 2);
  RangeSegment seg1(TEST_VEC_LEN / 2 + 1, TEST_VEC_LEN);

  IndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);

  double dtinit = 5.0;
  int itinit = 4;

  ReduceSum<reduction_type, double> dsum0(dtinit * 1.0);
  ReduceSum<reduction_type, int> isum1(itinit * 2);
  ReduceSum<reduction_type, double> dsum2(dtinit * 3.0);
  ReduceSum<reduction_type, int> isum3(itinit * 4);

  forall<IndexSet::ExecPolicy<seq_segit, execution_type> >(
      iset, [=] __device__(int i) {
        dsum0 += dvalue[i];
        isum1 += 2 * ivalue[i];
        dsum2 += 3 * dvalue[i];
        isum3 += 4 * ivalue[i];
      });

  double dbase_chk_val = dinit_val * double(iset.getLength());
  int ibase_chk_val = iinit_val * (iset.getLength());

  if (!equal(dsum0.get(), dbase_chk_val + (dtinit * 1.0))
      || !equal(isum1.get(), 2 * ibase_chk_val + (itinit * 2))
      || !equal(dsum2.get(), 3 * dbase_chk_val + (dtinit * 3.0))
      || !equal(isum3.get(), 4 * ibase_chk_val + (itinit * 4))) {
    cout << "\n TEST 2 <" <<  typeid(execution_type).name() << ", " << typeid(reduction_type).name() << "> FAILURE: tcount = " << tcount << endl;
    cout << setprecision(20)
         << "\tdsum0 = " << static_cast<double>(dsum0.get()) << " ("
         << dbase_chk_val + (dtinit * 1.0) << ") " << endl;
    cout << setprecision(20)
         << "\tisum1 = " << static_cast<double>(isum1.get()) << " ("
         << 2 * ibase_chk_val + (itinit * 2) << ") " << endl;
    cout << setprecision(20)
         << "\tdsum2 = " << static_cast<double>(dsum2.get()) << " ("
         << 3 * dbase_chk_val + (dtinit * 3.0) << ") " << endl;
    cout << setprecision(20)
         << "\tisum3 = " << static_cast<double>(isum3.get()) << " ("
         << 4 * ibase_chk_val + (itinit * 4) << ") " << endl;

  } else {
    s_ntests_passed++;
  }

}  // end test 2

//
// test 3 runs 4 reductions (2 int, 2 double) over disjoint chunks
//        of the array using an indexset with four range segments
//        not aligned with warp boundaries to check that reduction
//        mechanics don't depend on any sort of special indexing.
//
template<typename execution_type, typename reduction_type>
void test3()
{  // begin test 3

  s_ntests_run++;

  RangeSegment seg0(1, 1230);
  RangeSegment seg1(1237, 3385);
  RangeSegment seg2(4860, 10110);
  RangeSegment seg3(20490, 32003);

  IndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);
  iset.push_back(seg2);
  iset.push_back(seg3);

  double dtinit = 5.0;
  int itinit = 4;

  ReduceSum<reduction_type, double> dsum0(dtinit * 1.0);
  ReduceSum<reduction_type, int> isum1(itinit * 2);
  ReduceSum<reduction_type, double> dsum2(dtinit * 3.0);
  ReduceSum<reduction_type, int> isum3(itinit * 4);

  forall<IndexSet::ExecPolicy<seq_segit, execution_type > >(
      iset, [=] __device__(int i) {
        dsum0 += dvalue[i];
        isum1 += 2 * ivalue[i];
        dsum2 += 3 * dvalue[i];
        isum3 += 4 * ivalue[i];
      });

  double dbase_chk_val = dinit_val * double(iset.getLength());
  int ibase_chk_val = iinit_val * double(iset.getLength());

  if (!equal(double(dsum0), dbase_chk_val + (dtinit * 1.0))
      || !equal(int(isum1), 2 * ibase_chk_val + (itinit * 2))
      || !equal(double(dsum2), 3 * dbase_chk_val + (dtinit * 3.0))
      || !equal(int(isum3), 4 * ibase_chk_val + (itinit * 4))) {
    cout << "\n TEST 3 <" <<  typeid(execution_type).name() << ", " << typeid(reduction_type).name() << "> FAILURE: tcount = " << tcount << endl;
    cout << setprecision(20) << "\tdsum0 = " << static_cast<double>(dsum0)
         << " (" << dbase_chk_val + (dtinit * 1.0) << ") " << endl;
    cout << setprecision(20) << "\tisum1 = " << static_cast<double>(isum1)
         << " (" << 2 * ibase_chk_val + (itinit * 2) << ") " << endl;
    cout << setprecision(20) << "\tdsum2 = " << static_cast<double>(dsum2)
         << " (" << 3 * dbase_chk_val + (dtinit * 3.0) << ") " << endl;
    cout << setprecision(20) << "\tisum3 = " << static_cast<double>(isum3)
         << " (" << 4 * ibase_chk_val + (itinit * 4) << ") " << endl;

  } else {
    s_ntests_passed++;
  }

}  // end test 3


int main(int argc, char *argv[])
{
  cout << "\n Begin RAJA GPU ReduceSum tests!!! " << endl;

  //
  // Allocate and initialize managed data arrays
  //

  cudaMallocManaged((void **)&dvalue,
                    sizeof(double) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);
  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    dvalue[i] = dinit_val;
  }

  cudaMallocManaged((void **)&ivalue,
                    sizeof(int) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);
  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    ivalue[i] = iinit_val;
  }

  ////////////////////////////////////////////////////////////////////////////
  // Run 3 different sum reduction tests in a loop
  ////////////////////////////////////////////////////////////////////////////

  for (tcount = 0; tcount < test_repeat; ++tcount) {
    cout << "\t tcount = " << tcount << endl;

    test1<cuda_exec<block_size>, cuda_reduce<block_size>>();
    test1<cuda_exec<block_size>, cuda_reduce_atomic<block_size>>();
    test1<cuda_exec_async<block_size>, cuda_reduce<block_size>>();
    test1<cuda_exec_async<block_size>, cuda_reduce_atomic<block_size>>();
    test1<cuda_exec<block_size>, cuda_reduce_async<block_size>>();
    test1<cuda_exec<block_size>, cuda_reduce_async_atomic<block_size>>();
    test1<cuda_exec_async<block_size>, cuda_reduce_async<block_size>>();
    test1<cuda_exec_async<block_size>, cuda_reduce_async_atomic<block_size>>();

    ////////////////////////////////////////////////////////////////////////////

    test2<cuda_exec<block_size>, cuda_reduce<block_size>>();
    test2<cuda_exec<block_size>, cuda_reduce_atomic<block_size>>();
    test2<cuda_exec_async<block_size>, cuda_reduce<block_size>>();
    test2<cuda_exec_async<block_size>, cuda_reduce_atomic<block_size>>();
    test2<cuda_exec<block_size>, cuda_reduce_async<block_size>>();
    test2<cuda_exec<block_size>, cuda_reduce_async_atomic<block_size>>();
    test2<cuda_exec_async<block_size>, cuda_reduce_async<block_size>>();
    test2<cuda_exec_async<block_size>, cuda_reduce_async_atomic<block_size>>();

    ////////////////////////////////////////////////////////////////////////////

    test3<cuda_exec<block_size>, cuda_reduce<block_size>>();
    test3<cuda_exec<block_size>, cuda_reduce_atomic<block_size>>();
    test3<cuda_exec_async<block_size>, cuda_reduce<block_size>>();
    test3<cuda_exec_async<block_size>, cuda_reduce_atomic<block_size>>();
    test3<cuda_exec<block_size>, cuda_reduce_async<block_size>>();
    test3<cuda_exec<block_size>, cuda_reduce_async_atomic<block_size>>();
    test3<cuda_exec_async<block_size>, cuda_reduce_async<block_size>>();
    test3<cuda_exec_async<block_size>, cuda_reduce_async_atomic<block_size>>();

  }  // end test repeat loop

  ///
  /// Print total number of tests passed/run.
  ///
  cout << "\n Tests Passed / Tests Run = " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  RAJA::wait<cuda_wait>();

  cudaFree(dvalue);
  cudaFree(ivalue);

  return !(s_ntests_passed == s_ntests_run);
}
