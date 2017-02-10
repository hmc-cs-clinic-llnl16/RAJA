/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see RAJA/LICENSE.
 */

#include <iostream>
#include <string>
#include <vector>

#include "RAJA/RAJA.hxx"
#include "buildIndexSet.hxx"
#include "gtest/gtest.h"

class AgencyForallTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    expected = std::vector<double>(1000);
    actual = std::vector<double>(1000);

    y = std::vector<double>(1000);
    x = std::vector<double>(1000);

    for (auto i = 0; i < 1000; ++i) {
        y[i] = i;
        x[i] = 1000 - i;
    }
  }

  std::vector<double> expected;
  std::vector<double> actual;

  std::vector<double> y;
  std::vector<double> x;

  template <typename Policy>
  void forall_daxpy() {
      double a = 17.3;

      for (auto i = 0; i < 1000; ++i) {
        expected[i] = a*x[i] + y[i];
      }

      RAJA::forall<Policy>(
        0, 1000, [=](RAJA::Index_type i) {
          actual[i] = a*x[i] + y[i];
        });

      for (auto i = 0; i < 1000; ++i) {
        EXPECT_EQ(actual[i], expected[i]);
      }
  }

};

TEST_F(AgencyForallTest, daxpy_parallel)
{
    forall_daxpy<RAJA::experimental::agency_parallel_exec>();
}

TEST_F(AgencyForallTest, daxpy_sequential)
{
    forall_daxpy<RAJA::experimental::agency_sequential_exec>();
}

#if defined(RAJA_ENABLE_OPENMP)
TEST_F(AgencyForallTest, daxpy_omp_parallel)
{
    forall_daxpy<RAJA::experimental::agency_omp_parallel_exec>();
}
#endif // defined(RAJA_ENABLE_OPENMP)

class AgencyForallIcountTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        // Init arrays
        expected = std::vector<double>(1000);
        actual = std::vector<double>(1000);

        in_array = std::vector<double>(1000);
        for (auto i = 0; i < 1000; ++i) {
            in_array[i] = i;
        }
        
        // Init index sets
        for (auto ibuild = 0; ibuild < IndexSetBuildMethod::NumBuildMethods; ++ibuild) {
            buildIndexSet(indices, static_cast<IndexSetBuildMethod>(ibuild));
            RAJA::getIndices(is_indices[ibuild], indices[ibuild]);
        }    
    }

    RAJA::IndexSet indices[IndexSetBuildMethod::NumBuildMethods];
    RAJA::RAJAVec<RAJA::Index_type> is_indices[IndexSetBuildMethod::NumBuildMethods];

    std::vector<double> expected;
    std::vector<double> actual;

    std::vector<double> in_array;

    template <typename Policy>
    void forall_icount() 
    {
        for (auto ibuild = 0; ibuild < IndexSetBuildMethod::NumBuildMethods; ++ibuild) {
            for (size_t i = 0; i < is_indices[ibuild].size(); ++i) {
                expected[i] = in_array[is_indices[ibuild][i]] * in_array[is_indices[ibuild][i]];
            }

            RAJA::forall_Icount<Policy>(indices[ibuild], [=](RAJA::Index_type icount, RAJA::Index_type idx) {
                actual[icount] = in_array[idx] * in_array[idx];
            });

            for (auto i = 0; i < 1000; ++i) {
                EXPECT_EQ(actual[i], expected[i]);
            }
        }    
    }    
};

TEST_F(AgencyForallIcountTest, parallel)
{
    forall_icount<
        RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::experimental::agency_parallel_exec>
    >();
}

TEST_F(AgencyForallIcountTest, sequential)
{
    forall_icount<
        RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::experimental::agency_sequential_exec>
    >();
}

#if defined(RAJA_ENABLE_OPENMP)
TEST_F(AgencyForallIcountTest, omp_parallel)
{
    forall_icount<
        RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::experimental::agency_omp_parallel_exec>
    >();
}
#endif // defined(RAJA_ENABLE_OPENMP)

// TODO: Make sure we test all of: RangeSegment, Iterable, Container
// TODO: Test forallN
