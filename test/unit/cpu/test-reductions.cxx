//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read raja/README-license.txt.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"
#include <iostream>

#include <tuple>

template <typename T>
class ReductionConstructorTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(ReductionConstructorTest);

TYPED_TEST_P(ReductionConstructorTest, ReductionConstructor)
{
  using ReducePolicy = typename std::tuple_element<0, TypeParam>::type;
  using NumericType = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(0.0);
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(0.0);
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(0.0);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(0.0, 1);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(0.0, 1);

  ASSERT_EQ((NumericType) reduce_sum.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType) reduce_min.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType) reduce_max.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType) reduce_minloc.get(), (NumericType)(0.0));
  ASSERT_EQ((RAJA::Index_type) reduce_minloc.getLoc(), (RAJA::Index_type) 1);
  ASSERT_EQ((NumericType) reduce_maxloc.get(), (NumericType)(0.0));
  ASSERT_EQ((RAJA::Index_type) reduce_maxloc.getLoc(), (RAJA::Index_type) 1);
}

REGISTER_TYPED_TEST_CASE_P(ReductionConstructorTest, ReductionConstructor);

#if defined(RAJA_ENABLE_OPENMP)
using constructor_types = ::testing::Types<
    std::tuple<RAJA::seq_reduce, int>,
    std::tuple<RAJA::seq_reduce, float>,
    std::tuple<RAJA::seq_reduce, double>,
    std::tuple<RAJA::omp_reduce, int>,
    std::tuple<RAJA::omp_reduce, float>,
    std::tuple<RAJA::omp_reduce, double>,
    std::tuple<RAJA::omp_reduce_ordered, int>,
    std::tuple<RAJA::omp_reduce_ordered, float>,
    std::tuple<RAJA::omp_reduce_ordered, double>, 
    //Fix for seperate place for agency tests
    std::tuple<RAJA::agency_reduce, int>,
    std::tuple<RAJA::agency_reduce, float>,
    std::tuple<RAJA::agency_reduce, double> >;

#else
using constructor_types = ::testing::Types<
    std::tuple<RAJA::seq_reduce, int>,
    std::tuple<RAJA::seq_reduce, float>,
    std::tuple<RAJA::seq_reduce, double>,
    //Fix for seperate place for agency tests
    std::tuple<RAJA::agency_reduce, int>,
    std::tuple<RAJA::agency_reduce, float>,
    std::tuple<RAJA::agency_reduce, double> >;
#endif

INSTANTIATE_TYPED_TEST_CASE_P(ReduceBasicTests, ReductionConstructorTest, constructor_types);


template <typename TUPLE>
class ReductionCorrectnessTest : public ::testing::Test
{
 protected:
   virtual void SetUp()
   {
     array_length = 10200;

     array = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                 array_length * sizeof(double));

     for (int i = 1; i < array_length-1; ++i) {
       array[i] = (RAJA::Real_type) i;
     }
     array[0] = 0.0;
     array[array_length-1] = -1.0;

     sum = 0.0;
     min = array_length * 2;
     max = 0.0;
     minloc = -1;
     maxloc = -1;

     for (int i = 0; i < array_length; ++i) {
       RAJA::Real_type val = array[i];

       sum += val;

       if (val > max) {
         max = val;
         maxloc = i;
       }

       if (val < min) {
         min = val;
         minloc = i;
       }
     }
   }

   virtual void TearDown()
   {
     free(array);
   }

   RAJA::Real_ptr array;

   RAJA::Real_type max;
   RAJA::Real_type min;
   RAJA::Real_type sum;
   RAJA::Real_type maxloc;
   RAJA::Real_type minloc;

   RAJA::Index_type array_length;
 };
TYPED_TEST_CASE_P(ReductionCorrectnessTest);

TYPED_TEST_P(ReductionCorrectnessTest, ReduceSum)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  //using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, double> sum_reducer(0.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length), [=] (int i) {
    sum_reducer += this->array[i];
  });

  double raja_sum = (double) sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMin)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  //using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMin<ReducePolicy, double> min_reducer(1024.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length), [=] (int i) {
    min_reducer.min(this->array[i]);
  });

  double raja_min = (double) min_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMax)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  //using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMax<ReducePolicy, double> max_reducer(0.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length), [=] (int i) {
    max_reducer.max(this->array[i]);
  });

  double raja_max = (double) max_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMinLoc)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  //using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMinLoc<ReducePolicy, double> minloc_reducer(1024.0, 0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length), [=] (int i) {
    minloc_reducer.minloc(this->array[i], i);
  });

  double raja_min = (double) minloc_reducer.get();
  RAJA::Index_type raja_loc = minloc_reducer.getLoc();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minloc, raja_loc);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMaxLoc)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  //using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMaxLoc<ReducePolicy, double> maxloc_reducer(0.0, -1);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length), [=] (int i) {
    maxloc_reducer.maxloc(this->array[i], i);
  });

  double raja_max = (double) maxloc_reducer.get();
  RAJA::Index_type raja_loc = maxloc_reducer.getLoc();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxloc, raja_loc);
}

REGISTER_TYPED_TEST_CASE_P(ReductionCorrectnessTest, ReduceSum, ReduceMin,
                           ReduceMax, ReduceMinLoc, ReduceMaxLoc);

#if defined(RAJA_ENABLE_OPENMP)
using types = ::testing::Types<
    std::tuple<RAJA::seq_exec, RAJA::seq_reduce>,
    std::tuple<RAJA::simd_exec, RAJA::seq_reduce>,
    std::tuple<RAJA::omp_parallel_for_exec, RAJA::omp_reduce>,
    std::tuple<RAJA::omp_parallel_for_exec, RAJA::omp_reduce_ordered>,
    //add logic for including agency
    std::tuple<RAJA::agency_sequential_exec, RAJA::agency_reduce>,
    std::tuple<RAJA::agency_parallel_exec, RAJA::agency_reduce>
>;
#else
using types = ::testing::Types<
    std::tuple<RAJA::seq_exec, RAJA::seq_reduce>,
    std::tuple<RAJA::simd_exec, RAJA::seq_reduce>,
    //add logic for including agency
    std::tuple<RAJA::agency_sequential_exec, RAJA::agency_reduce>,
    std::tuple<RAJA::agency_parallel_exec, RAJA::agency_reduce>
>;
#endif

INSTANTIATE_TYPED_TEST_CASE_P(Reduce, ReductionCorrectnessTest, types);

template <typename TUPLE>
class NestedReductionCorrectnessTest : public ::testing::Test
{
 protected:
   virtual void SetUp()
   {
     x_size = 16;
     y_size = 16;
     z_size = 16;

     array = 
       RAJA::allocate_aligned_type<double>(
           RAJA::DATA_ALIGN, x_size * y_size * z_size * sizeof(double));

     const double val = 4.0/(x_size * y_size * z_size);

     for (int i = 0; i < (x_size * y_size * z_size); ++i) {
           array[i] = (RAJA::Real_type) val;
     }

     sum = 4.0;
   }

   virtual void TearDown()
   {
     free(array);
   }

   RAJA::Real_ptr array;

   RAJA::Real_type sum;

   RAJA::Index_type x_size;
   RAJA::Index_type y_size;
   RAJA::Index_type z_size;
 };
TYPED_TEST_CASE_P(NestedReductionCorrectnessTest);

TYPED_TEST_P(NestedReductionCorrectnessTest, NestedReduceSum)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, double> sum_reducer(0.0);

  RAJA::View<double, RAJA::Layout<RAJA::Index_type, RAJA::PERM_IJK, RAJA::Index_type, RAJA::Index_type, RAJA::Index_type> > view(
      this->array, this->x_size, this->y_size, this->z_size);

  RAJA::forallN<ExecPolicy>(
      RAJA::RangeSegment(0, this->x_size), 
      RAJA::RangeSegment(0, this->y_size), 
      RAJA::RangeSegment(0, this->z_size), 
      [=] (int i, int j, int k) {
        sum_reducer += view(i, j, k);//sum_reducer += view(i, j, k);
  });

  double raja_sum = (double) sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

REGISTER_TYPED_TEST_CASE_P(NestedReductionCorrectnessTest, NestedReduceSum);

#if defined(RAJA_ENABLE_OPENMP)
using nested_types = ::testing::Types<
  std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
        RAJA::seq_exec,
        RAJA::seq_exec> >, RAJA::seq_reduce>, 
  std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_collapse_nowait_exec,
        RAJA::omp_collapse_nowait_exec,
        RAJA::omp_collapse_nowait_exec>,
        RAJA::OMP_Parallel<> >, RAJA::omp_reduce>, 
  std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,
        RAJA::seq_exec,
        RAJA::seq_exec> >, RAJA::omp_reduce>, 
  std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_collapse_nowait_exec,
        RAJA::omp_collapse_nowait_exec,
        RAJA::omp_collapse_nowait_exec>,
        RAJA::OMP_Parallel<> >, RAJA::omp_reduce_ordered>
  //add agency stuff
  // std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::agency_parallel_exec
  //       RAJA::agency_parallel_exec,
  //       RAJA::agency_parallel_exec> >, RAJA::agency_reduce>, 
  // std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::agency_sequential_exec
  //       RAJA::agency_sequential_exec,
  //       RAJA::agency_sequential_exec> >, RAJA::agency_reduce>
>;
#else
using nested_types = ::testing::Types<
  std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
        RAJA::seq_exec,
        RAJA::seq_exec> >, RAJA::seq_reduce>
  //add agency stuff
  // std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::agency_parallel_exec
  //       RAJA::agency_parallel_exec,
  //       RAJA::agency_parallel_exec> >, RAJA::agency_reduce>, 
  // std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::agency_sequential_exec
  //       RAJA::agency_sequential_exec,
  //       RAJA::agency_sequential_exec> >, RAJA::agency_reduce>
>;
#endif

INSTANTIATE_TYPED_TEST_CASE_P(NestedReduce, NestedReductionCorrectnessTest, nested_types);
