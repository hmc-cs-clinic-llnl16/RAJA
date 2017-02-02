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
#include <random>
#include <vector>

#include "RAJA/RAJA.hxx"
#include "gtest/gtest.h"

class AgencyTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    std::mt19937 gen(std::random_device());
    std::uniform_real_distribution<double> dist(0, 5);
    expected = std::vector<double>(1000);
    actual = std::vector<double>(1000);

    y = std::vector<double>(1000);
    x = std::vector<double>(1000);

    for (auto i = 0; i < 1000; ++i) {
        y[i] = dist(gen);
        x[i] = dist(gen);
    }
  }

  std::vector<double> expected;
  std::vector<double> actual;

  std::vector<double> y;
  std::vector<double> x;
};

TEST_F(AgencyTest, forall_daxpy)
{
  double a = 17.3;

  // Do expected result
  for (auto i = 0; i < 1000; ++i) {
    expected[i] = a*x[i] + y[i];
  }

  // Do actual result
  RAJA::forall<RAJA::agency_parallel_exec>(
    0, 1000, [=](RAJA::Index_type i) {
      actual[i] = a*x[i] + y[i];
    });

  // Validate_result
  for (auto i = 0; i < 1000; ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
}
