/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

#include <cfloat>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>

#include "RAJA/RAJA.hxx"

#define TEST_VEC_LEN 1024 * 1024 * 11

using namespace RAJA;
using namespace std;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

const int test_repeat = 10;

//
// Allocate and initialize managed data array
//
__managed__ double *dvalue;

///
/// Define thread block size for CUDA exec policy
///
const size_t block_size = 256;

// current running min value
double dcurrentMin = DBL_MAX;

// for setting random min values in arrays
random_device rd;
mt19937 mt(rd());
uniform_real_distribution<double> dist(-10, 10);
uniform_real_distribution<double> dist2(0, TEST_VEC_LEN - 1);

int tcount = 0;


//
// test 1 runs 3 reductions over a range multiple times to check
//        that reduction value can be retrieved and then subsequent
//        reductions can be run with the same reduction objects.
//        Also exercises the get() function call
template<typename execution_type, typename reduction_type>
void test1()
{  // begin test 1

  double BIG_MIN = -500.0;
  ReduceMin<reduction_type, double> dmin0(DBL_MAX);
  ReduceMin<reduction_type, double> dmin1(DBL_MAX);
  ReduceMin<reduction_type, double> dmin2(BIG_MIN);

  int loops = 16;
  for (int k = 0; k < loops; k++) {
    s_ntests_run++;

    RAJA::wait<cuda_wait>();

    double droll = dist(mt);
    int index = int(dist2(mt));
    dvalue[index] = droll;
    dcurrentMin = RAJA_MIN(dcurrentMin, dvalue[index]);

    forall<execution_type >(0, TEST_VEC_LEN, [=] __device__(int i) {
      dmin0.min(dvalue[i]);
      dmin1.min(2 * dvalue[i]);
      dmin2.min(dvalue[i]);
    });

    if (dmin0.get() != dcurrentMin || dmin1.get() != 2 * dcurrentMin
        || dmin2.get() != BIG_MIN) {
      cout << "\n TEST 1 FAILURE: tcount, k = " << tcount << " , " << k
           << endl;
      cout << "  droll = " << droll << endl;
      cout << "\tdmin0 = " << static_cast<double>(dmin0.get()) << " ("
           << dcurrentMin << ") " << endl;
      cout << "\tdmin1 = " << static_cast<double>(dmin1.get()) << " ("
           << 2 * dcurrentMin << ") " << endl;
      cout << "\tdmin2 = " << static_cast<double>(dmin2.get()) << " ("
           << BIG_MIN << ") " << endl;
    } else {
      s_ntests_passed++;
    }
  }

}  // end test 1

////////////////////////////////////////////////////////////////////////////

//
// test 2 runs 2 reductions over complete array using an indexset
//        with two range segments to check reduction object state
//        is maintained properly across kernel invocations.
//
template<typename execution_type, typename reduction_type>
void test2()
{  // begin test 2

  s_ntests_run++;

  RangeSegment seg0(0, TEST_VEC_LEN / 2);
  RangeSegment seg1(TEST_VEC_LEN / 2 + 1, TEST_VEC_LEN);

  IndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);

  ReduceMin<reduction_type, double> dmin0(DBL_MAX);
  ReduceMin<reduction_type, double> dmin1(DBL_MAX);

  int index = int(dist2(mt));

  RAJA::wait<cuda_wait>();

  double droll = dist(mt);
  dvalue[index] = droll;

  dcurrentMin = RAJA_MIN(dcurrentMin, dvalue[index]);

  forall<IndexSet::ExecPolicy<seq_segit, execution_type > >(
      iset, [=] __device__(int i) {
        dmin0.min(dvalue[i]);
        dmin1.min(2 * dvalue[i]);
      });

  if (double(dmin0) != dcurrentMin || double(dmin1) != 2 * dcurrentMin) {
    cout << "\n TEST 2 FAILURE: tcount = " << tcount << endl;
    cout << "   droll = " << droll << endl;
    cout << "\tdmin0 = " << static_cast<double>(dmin0) << " ("
         << dcurrentMin << ") " << endl;
    cout << "\tdmin1 = " << static_cast<double>(dmin1) << " ("
         << 2 * dcurrentMin << ") " << endl;
  } else {
    s_ntests_passed++;
  }

}  // end test 2

///////////////////////////////////////////////////////////////////////

//
// test 3 runs 2 reductions over disjoint chunks of the array using
//        an indexset with four range segments not aligned with
//        warp boundaries to check that reduction mechanics don't
//        depend on any sort of special indexing.
//
template<typename execution_type, typename reduction_type>
void test3()
{  // begin test 3

  s_ntests_run++;

  RAJA::wait<cuda_wait>();

  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    dvalue[i] = DBL_MAX;
  }
  dcurrentMin = DBL_MAX;

  RangeSegment seg0(1, 1230);
  RangeSegment seg1(1237, 3385);
  RangeSegment seg2(4860, 10110);
  RangeSegment seg3(20490, 32003);

  IndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);
  iset.push_back(seg2);
  iset.push_back(seg3);

  ReduceMin<reduction_type, double> dmin0(DBL_MAX);
  ReduceMin<reduction_type, double> dmin1(DBL_MAX);

  // pick an index in one of the segments
  int index = 897;                     // seg 0
  if (tcount % 2 == 0) index = 1297;   // seg 1
  if (tcount % 3 == 0) index = 7853;   // seg 2
  if (tcount % 4 == 0) index = 29457;  // seg 3

  RAJA::wait<cuda_wait>();

  double droll = dist(mt);
  dvalue[index] = droll;

  dcurrentMin = RAJA_MIN(dcurrentMin, dvalue[index]);

  forall<IndexSet::ExecPolicy<seq_segit, execution_type > >(
      iset, [=] __device__(int i) {
        dmin0.min(dvalue[i]);
        dmin1.min(2 * dvalue[i]);
      });

  if (double(dmin0) != dcurrentMin || double(dmin1) != 2 * dcurrentMin) {
    cout << "\n TEST 3 FAILURE: tcount = " << tcount << endl;
    cout << "   droll = " << droll << endl;
    cout << "\tdmin0 = " << static_cast<double>(dmin0) << " ("
         << dcurrentMin << ") " << endl;
    cout << "\tdmin1 = " << static_cast<double>(dmin1) << " ("
         << 2 * dcurrentMin << ") " << endl;
  } else {
    s_ntests_passed++;
  }

}  // end test 3

int main(int argc, char *argv[])
{
  cout << "\n Begin RAJA GPU ReduceMin tests!!! " << endl;

  cudaMallocManaged((void **)&dvalue,
                    sizeof(double) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);
  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    dvalue[i] = DBL_MAX;
  }

  ////////////////////////////////////////////////////////////////////////////
  // Run 3 different min reduction tests in a loop
  ////////////////////////////////////////////////////////////////////////////

  for (tcount = 0; tcount < test_repeat; ++tcount) {
    cout << "\t tcount = " << tcount << endl;

    test1<cuda_exec<block_size>, cuda_reduce<block_size>>();

    test1<cuda_exec_async<block_size>, cuda_reduce<block_size>>();

    test1<cuda_exec<block_size>, cuda_reduce_async<block_size>>();

    test1<cuda_exec_async<block_size>, cuda_reduce_async<block_size>>();


    ////////////////////////////////////////////////////////////////////////////

    test2<cuda_exec<block_size>, cuda_reduce<block_size>>();

    test2<cuda_exec_async<block_size>, cuda_reduce<block_size>>();

    test2<cuda_exec<block_size>, cuda_reduce_async<block_size>>();

    test2<cuda_exec_async<block_size>, cuda_reduce_async<block_size>>();


    ////////////////////////////////////////////////////////////////////////////

    test3<cuda_exec<block_size>, cuda_reduce<block_size>>();

    test3<cuda_exec_async<block_size>, cuda_reduce<block_size>>();

    test3<cuda_exec<block_size>, cuda_reduce_async<block_size>>();

    test3<cuda_exec_async<block_size>, cuda_reduce_async<block_size>>();


  }  // end test repeat loop

  ///
  /// Print total number of tests passed/run.
  ///
  cout << "\n Tests Passed / Tests Run = " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  RAJA::wait<cuda_wait>();

  cudaFree(dvalue);

  cout << "\n RAJA GPU ReduceMin tests DONE!!! " << endl;

  return !(s_ntests_passed == s_ntests_run);
}
