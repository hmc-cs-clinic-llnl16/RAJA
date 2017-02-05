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
#include "gtest/gtest.h"

class AgencyTest : public ::testing::Test
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

      // Do expected result
      for (auto i = 0; i < 1000; ++i) {
        expected[i] = a*x[i] + y[i];
      }

      // Do actual result
      RAJA::forall<Policy>(
        0, 1000, [=](RAJA::Index_type i) {
          actual[i] = a*x[i] + y[i];
        });

      // Validate_result
      for (auto i = 0; i < 1000; ++i) {
        EXPECT_EQ(actual[i], expected[i]);
      }
  }
};

TEST_F(AgencyTest, forall_daxpy_agency_parallel)
{
    forall_daxpy<RAJA::experimental::agency_parallel_exec>();
}

TEST_F(AgencyTest, forall_daxpy_agency_sequential)
{
    forall_daxpy<RAJA::experimental::agency_sequential_exec>();
}

TEST_F(AgencyTest, forall_daxpy_agency_omp_parallel)
{
    forall_daxpy<RAJA::experimental::agency_omp_parallel_exec>();
}

TEST_F(AgencyTest, forall_daxpy_agency_omp_sequential)
{
    forall_daxpy<RAJA::experimental::agency_omp_sequential_exec>();
}
