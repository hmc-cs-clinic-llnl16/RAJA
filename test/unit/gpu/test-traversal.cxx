/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see RAJA/LICENSE.
 */

#include <cfloat>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "RAJA/RAJA.hxx"

using namespace RAJA;
using namespace std;

#include "Compare.hxx"

#include <gtest/gtest.h>

template <typename T>
class BaseForallTest : public ::testing::Test
{
protected:
  Real_ptr test_array;
  Real_ptr ref_array;
  Real_ptr parent;
  Index_type array_length; 

  RAJAVec<Index_type> lindices;

  IndexSet iset;
  RAJAVec<Index_type> is_indices;

  virtual void SetUpIndexSet()
  {
      Index_type idx = 0;
      while (lindices.size() < 10000) {
        double dval = rand();
        if (dval > 0.3) {
          lindices.push_back(idx);
        }
        idx++;
      }

      //
      // Construct index set with mix of Range and List segments.
      //
      Index_type rbeg;
      Index_type rend;
      Index_type last_idx;
      Index_type lseg_len = lindices.size();
      RAJAVec<Index_type> lseg(lseg_len);
      std::vector<Index_type> lseg_vec(lseg_len);

      // Create empty Range segment
      rbeg = 1;
      rend = 1;
      iset.push_back(RangeSegment(rbeg, rend));
      last_idx = rend;

      // Create Range segment
      rbeg = 1;
      rend = 15782;
      iset.push_back(RangeSegment(rbeg, rend));
      last_idx = rend;

      // Create List segment
      for (Index_type i = 0; i < lseg_len; ++i) {
        lseg[i] = lindices[i] + last_idx + 3;
      }
      iset.push_back(ListSegment(&lseg[0], lseg_len));
      last_idx = lseg[lseg_len - 1];

      // Create List segment using alternate ctor
      for (Index_type i = 0; i < lseg_len; ++i) {
        lseg_vec[i] = lindices[i] + last_idx + 3;
      }
      iset.push_back(ListSegment(lseg_vec));
      last_idx = lseg_vec[lseg_len - 1];

      // Create Range segment
      rbeg = last_idx + 16;
      rend = rbeg + 20490;
      iset.push_back(RangeSegment(rbeg, rend));
      last_idx = rend;

      // Create Range segment
      rbeg = last_idx + 4;
      rend = rbeg + 27595;
      iset.push_back(RangeSegment(rbeg, rend));
      last_idx = rend;

      // Create List segment
      for (Index_type i = 0; i < lseg_len; ++i) {
        lseg[i] = lindices[i] + last_idx + 5;
      }
      iset.push_back(ListSegment(&lseg[0], lseg_len));
      last_idx = lseg[lseg_len - 1];

      // Create Range segment
      rbeg = last_idx + 1;
      rend = rbeg + 32003;
      iset.push_back(RangeSegment(rbeg, rend));
      last_idx = rend;

      // Create List segment using alternate ctor
      for (Index_type i = 0; i < lseg_len; ++i) {
        lseg_vec[i] = lindices[i] + last_idx + 7;
      }
      iset.push_back(ListSegment(lseg_vec));
      last_idx = lseg_vec[lseg_len - 1];

      //
      // Collect actual indices in index set for testing.
      //
      getIndices(is_indices, iset);

      array_length = last_idx + 1;
  }

  virtual void SetUp()
  {
      SetUpIndexSet();

      cudaMallocManaged((void **)&parent,
                        sizeof(Real_type) * array_length,
                        cudaMemAttachGlobal);
      for (Index_type i = 0; i < array_length; ++i) {
        parent[i] = static_cast<Real_type>(rand() % 65536);
      }

      cudaMallocManaged((void **)&test_array,
                        sizeof(Real_type) * array_length,
                        cudaMemAttachGlobal);
      cudaMallocManaged((void **)&ref_array,
                        sizeof(Real_type) * array_length,
                        cudaMemAttachGlobal);
      cudaMemset(test_array, 0, sizeof(Real_type) * array_length);
      cudaMemset(ref_array, 0, sizeof(Real_type) * array_length);
  }

  virtual void TearDown()
  {
      cudaFree(parent);
      cudaFree(ref_array);
      cudaFree(test_array);
  }
};

template <typename T>
class SimpleForallTest : protected BaseForallTest<T>
{
protected:
  virtual void SetUp()
  {
    BaseForallTest<T>::SetUp();

    for (Index_type i = 0; i < array_length; ++i) {
        ref_array[i] = parent[i] * parent[i];
    }
  }
};

TYPED_TEST_CASE_P(SimpleForallTest);
TYPED_TEST_P(SimpleForallTest, SimpleForall)
{
    using Policy = TypeParam;

    // Trivial case as sanity check
    forall<Policy>(0, 0, 
                   [=] __device__ (Index_type idx) {
                       this->test_array[idx] = this->parent[idx] * this->parent[idx];
                   });    

    EXPECT_TRUE(array_equal(this->ref_array, this->test_array, 0));

    cudaMemset(this->test_array, 0, sizeof(Real_type) * this->array_length);

    // Simple case
    forall<Policy>(0, this->array_length, 
                   [=] __device__ (Index_type idx) {
                       this->test_array[idx] = this->parent[idx] * this->parent[idx];
                   });    

    EXPECT_TRUE(array_equal(this->ref_array, this->test_array, this->array_length));
}                   

TYPED_TEST_P(SimpleForallTest, SimpleForallIcount)
{
    using Policy = TypeParam;

    forall_Icount<Policy >(
        0, array_length, 0, 
        [=] __device__(Index_type icount, Index_type idx) {
            this->test_array[icount] = this->parent[idx] * this->parent[idx];
        });

    EXPECT_TRUE(array_equal(this->ref_array, this->test_array, this->array_length));
}                   

REGISTER_TYPED_TEST_CASE_P(SimpleForallTest, SimpleForall, SimpleForallIcount);

INSTANTIATE_TYPED_TEST_CASE_P(CUDA, SimpleForallTest, cuda_exec<256>);
#if defined(RAJA_ENABLE_AGENCY)
INSTANTIATE_TYPED_TEST_CASE_P(Agency, SimpleForallTest, agency_cuda_exec);
#endif

template <typename T>
class MultiForallTest : public BaseForallTest<T>
{
protected:
  virtual void SetUp()
  {
      BaseForallTest<T>::SetUp();

      for (Index_type i = 0; i < is_indices.size(); ++i) {
          ref_array[is_indices[i]] = parent[is_indices[i]] * parent[is_indices[i]];
      }
  }
};

TYPED_TEST_CASE_P(MultiForallTest);
TYPED_TEST_P(MultiForallTest, MultiSegmentForall)
{
    using Policy = IndexSet::ExecPolicy<seq_segit, TypeParam>;

    forall<Policy>(this->iset, 
                   [=] __device__(Index_type idx) {
                       this->test_array[idx] = this->parent[idx] * this->parent[idx];
                   });

    EXPECT_TRUE(array_equal(this->ref_array, this->test_array, this->array_length));
}                   

REGISTER_TYPED_TEST_CASE_P(MultiForallTest, MultiSegmentForall);

INSTANTIATE_TYPED_TEST_CASE_P(CUDA, MultiForallTest, cuda_exec<256>);
#if defined(RAJA_ENABLE_AGENCY)
INSTANTIATE_TYPED_TEST_CASE_P(Agency, MultiForallTest, agency_cuda_exec);
#endif

template <typename T>
class MultiForallIcountTest : public BaseForallTest<T>
{
protected:
  virtual void SetUp()
  {
      BaseForallTest<T>::SetUp();

      for (Index_type i = 0; i < is_indices.size(); ++i) {
          ref_array[i] = parent[is_indices[i]] * parent[is_indices[i]];
      }
  }
};

TYPED_TEST_CASE_P(MultiForallIcountTest);
TYPED_TEST_P(MultiForallIcountTest, MultiSegmentForallIcount)
{
    using Policy = IndexSet::ExecPolicy<seq_segit, TypeParam>;

    forall_Icount<Policy>(
        iset, 
        [=] __device__ (Index_type icount, Index_type idx) {
            this->test_array[icount] = this->parent[idx] * this->parent[idx];
         });

    EXPECT_TRUE(array_equal(this->ref_array, this->test_array, this->array_length));
}                   

REGISTER_TYPED_TEST_CASE_P(MultiForallIcountTest, MultiSegmentForallIcount);

INSTANTIATE_TYPED_TEST_CASE_P(CUDA, MultiForallIcountTest, cuda_exec<256>);
#if defined(RAJA_ENABLE_AGENCY)
INSTANTIATE_TYPED_TEST_CASE_P(Agency, MultiForallIcountTest, agency_cuda_exec);
#endif

