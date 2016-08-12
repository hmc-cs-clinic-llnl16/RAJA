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

#define TEST_VEC_LEN 1024 * 1024 * 13

typedef struct {
  double val;
  int idx;
} minloc_t;

using namespace RAJA;
using namespace std;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

const int test_repeat = 10;

__managed__ double *dvalue;

///
/// Define thread block size for CUDA exec policy
///
const size_t block_size = 256;

// current running min value

minloc_t dcurrentMin;

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
//        Also exercises the get function call
template<typename execution_type, typename reduction_type>
void test1()
{  // begin test 1

  double BIG_MIN = -500.0;
  ReduceMinLoc<reduction_type, double> dmin0(DBL_MAX, -1);
  ReduceMinLoc<reduction_type, double> dmin1(DBL_MAX, 1);
  ReduceMinLoc<reduction_type, double> dmin2(BIG_MIN, -1);

  int loops = 16;
  for (int k = 0; k < loops; k++) {
    s_ntests_run++;

    RAJA::wait<cuda_wait>();

    double droll = dist(mt);
    int index = int(dist2(mt));
    dvalue[index] = droll;

    minloc_t lmin = {droll, index};
    dvalue[index] = droll;
    dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);
    // printf("droll %lf : dcurrentMin %lf : index
    // %d\n",droll,dcurrentMin,index);
    forall<execution_type >(0, TEST_VEC_LEN, [=] __device__(int i) {
      dmin0.minloc(dvalue[i], i);
      dmin1.minloc(2 * dvalue[i], i);
      dmin2.minloc(dvalue[i], i);
    });

    if (dmin0.get() != dcurrentMin.val || dmin1.get() != 2 * dcurrentMin.val
        || dmin2.get() != BIG_MIN
        || dmin0.getLoc() != dcurrentMin.idx
        || dmin1.getLoc() != dcurrentMin.idx) {
      cout << "\n TEST 1 FAILURE: tcount, k = " << tcount << " , " << k
           << endl;
      cout << "  droll = " << droll << endl;
      cout << "\tdmin0 = " << static_cast<double>(dmin0.get()) << ", " << dmin0.getLoc() << " ("
           << dcurrentMin.val << ", " << dcurrentMin.idx << ") " << endl;
      cout << "\tdmin1 = " << static_cast<double>(dmin1.get()) << ", " << dmin1.getLoc() << " ("
           << 2 * dcurrentMin.val << ", " << dcurrentMin.idx << ") " << endl;
      cout << "\tdmin2 = " << static_cast<double>(dmin2.get()) << ", " << dmin2.getLoc() << " ("
           << BIG_MIN << ", " << -1 << ") " << endl;
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

  ReduceMinLoc<reduction_type, double> dmin0(DBL_MAX, -1);
  ReduceMinLoc<reduction_type, double> dmin1(DBL_MAX, -1);

  int index = int(dist2(mt));

  RAJA::wait<cuda_wait>();

  double droll = dist(mt);
  dvalue[index] = droll;

  minloc_t lmin = {droll, index};
  dvalue[index] = droll;
  dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);

  forall<IndexSet::ExecPolicy<seq_segit, execution_type > >(
      iset, [=] __device__(int i) {
        dmin0.minloc(dvalue[i], i);
        dmin1.minloc(2 * dvalue[i], i);
      });

  if (double(dmin0) != dcurrentMin.val
      || double(dmin1) != 2 * dcurrentMin.val
      || dmin0.getLoc() != dcurrentMin.idx
      || dmin1.getLoc() != dcurrentMin.idx) {
    cout << "\n TEST 2 FAILURE: tcount = " << tcount << endl;
    cout << "   droll = " << droll << endl;
    cout << "\tdmin0 = " << static_cast<double>(dmin0) << ", " << dmin0.getLoc() << " ("
         << dcurrentMin.val << ", " << dcurrentMin.idx << ") " << endl;
    cout << "\tdmin1 = " << static_cast<double>(dmin1) << ", " << dmin1.getLoc() << " ("
         << 2 * dcurrentMin.val << ", " << dcurrentMin.idx << ") " << endl;
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

  dcurrentMin.val = DBL_MAX;
  dcurrentMin.idx = -1;

  RangeSegment seg0(1, 1230);
  RangeSegment seg1(1237, 3385);
  RangeSegment seg2(4860, 10110);
  RangeSegment seg3(20490, 32003);

  IndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);
  iset.push_back(seg2);
  iset.push_back(seg3);

  ReduceMinLoc<reduction_type, double> dmin0(DBL_MAX, -1);
  ReduceMinLoc<reduction_type, double> dmin1(DBL_MAX, -1);

  // pick an index in one of the segments
  int index = 897;                     // seg 0
  if (tcount % 2 == 0) index = 1297;   // seg 1
  if (tcount % 3 == 0) index = 7853;   // seg 2
  if (tcount % 4 == 0) index = 29457;  // seg 3

  RAJA::wait<cuda_wait>();

  double droll = dist(mt);
  dvalue[index] = droll;

  minloc_t lmin = {droll, index};
  dvalue[index] = droll;
  dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);

  forall<IndexSet::ExecPolicy<seq_segit, execution_type > >(
      iset, [=] __device__(int i) {
        dmin0.minloc(dvalue[i], i);
        dmin1.minloc(2 * dvalue[i], i);
      });

  if (double(dmin0) != dcurrentMin.val
      || double(dmin1) != 2 * dcurrentMin.val
      || dmin0.getLoc() != dcurrentMin.idx
      || dmin1.getLoc() != dcurrentMin.idx) {
    cout << "\n TEST 3 FAILURE: tcount = " << tcount << endl;
    cout << "   droll = " << droll << endl;
    cout << "\tdmin0 = " << static_cast<double>(dmin0) << ", " << dmin0.getLoc() << " ("
         << dcurrentMin.val << ", " << dcurrentMin.idx << ") " << endl;
    cout << "\tdmin1 = " << static_cast<double>(dmin1) << ", " << dmin1.getLoc() << " ("
         << 2 * dcurrentMin.val << ", " << dcurrentMin.idx << ") " << endl;
  } else {
    s_ntests_passed++;
  }

}  // end test 3

///////////////////////////////////////////////////////////////////////

//
// test 4 runs 1 reductions over a chunk that causes only some threads
// of the last block to run. This test checks if the reduction is 
// correct despite the block that does the final reduction having less
// than the full amount of threads running.  This test may fail only
// intermittently.
//
template<typename execution_type, typename reduction_type>
void test4()
{  // begin test 4

  s_ntests_run++;

  RAJA::wait<cuda_wait>();

  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    dvalue[i] = DBL_MAX;
  }

  dcurrentMin.val = DBL_MAX;
  dcurrentMin.idx = -1;

  RangeSegment seg0(1, block_size * block_size - 1 + 1);

  IndexSet iset;
  iset.push_back(seg0);

  ReduceMinLoc<reduction_type, double> dmin0(DBL_MAX, -1);

  // pick an index in one of the segments
  int index = block_size * block_size - 2;     // seg 0

  RAJA::wait<cuda_wait>();

  double droll = dist(mt);
  dvalue[index] = droll;

  minloc_t lmin = {droll, index};
  dvalue[index] = droll;
  dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);

  forall<IndexSet::ExecPolicy<seq_segit, execution_type > >(
      iset, [=] __device__(int i) {
        dmin0.minloc(dvalue[i], i);
      });

  if (double(dmin0) != dcurrentMin.val
      || dmin0.getLoc() != dcurrentMin.idx) {
    cout << "\n TEST 4 FAILURE: tcount = " << tcount << endl;
    cout << "   droll = " << droll << endl;
    cout << "\tdmin0 = " << static_cast<double>(dmin0) << ", " << dmin0.getLoc() << " ("
         << dcurrentMin.val << ", " << dcurrentMin.idx << ") " << endl;
  } else {
    s_ntests_passed++;
  }

}  // end test 4

int main(int argc, char *argv[])
{
  cout << "\n Begin RAJA GPU ReduceMinLoc tests!!! " << endl;

  //
  // Allocate and initialize managed data array
  //

  cudaMallocManaged((void **)&dvalue,
                    sizeof(double) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);
  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    dvalue[i] = DBL_MAX;
  }

  ////////////////////////////////////////////////////////////////////////////
  // Run 3 different min reduction tests in a loop
  ////////////////////////////////////////////////////////////////////////////

  dcurrentMin.val = DBL_MAX;
  dcurrentMin.idx = -1;

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


    ////////////////////////////////////////////////////////////////////////////

    test4<cuda_exec<block_size>, cuda_reduce<block_size>>();

    test4<cuda_exec_async<block_size>, cuda_reduce<block_size>>();

    test4<cuda_exec<block_size>, cuda_reduce_async<block_size>>();

    test4<cuda_exec_async<block_size>, cuda_reduce_async<block_size>>();

  }    // end test repeat loop

  ///
  /// Print total number of tests passed/run.
  ///
  cout << "\n Tests Passed / Tests Run = " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  RAJA::wait<cuda_wait>();

  cudaFree(dvalue);

  cout << "\n RAJA GPU ReduceMinLoc tests DONE!!! " << endl;

  return !(s_ntests_passed == s_ntests_run);
}
