#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"

#include "buildIndexSet.hxx"

#include <set>

class IndexSetTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      buildIndexSet(index_sets_, static_cast<IndexSetBuildMethod>(ibuild));
    }

    getIndices(is_indices, index_sets_[0]);
  }

  RAJA::RAJAVec<RAJA::Index_type> is_indices;
  RAJA::IndexSet index_sets_[NumBuildMethods];
};


TEST_F(IndexSetTest, IndexSetEquality)
{
  for (unsigned ibuild = 1; ibuild < NumBuildMethods; ++ibuild) {
    EXPECT_EQ(index_sets_[ibuild], index_sets_[1]);
  }
}

// TODO: tests for adding other invalid types
TEST_F(IndexSetTest, InvalidSegments)
{
  RAJA::RangeStrideSegment rs_segment(0, 4, 2);

  EXPECT_NE(true, index_sets_[0].isValidSegmentType(&rs_segment));
  EXPECT_NE(true, index_sets_[0].push_back(rs_segment));
  EXPECT_NE(true, index_sets_[0].push_back_nocopy(&rs_segment));
}


template <typename T>
class DependentIndexSetTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(DependentIndexSetTest);

TYPED_TEST_P(DependentIndexSetTest, ForwardDependentIndexSets)
{
  using ExecPolicy = TypeParam;

  const int numSegments = 10;
  const int sizeOfSegment = 1;
  RAJA::IndexSet indexSet;
  
  for (int i = 0; i < numSegments; ++i) {
    indexSet.push_back(RAJA::RangeSegment(i * sizeOfSegment, i * sizeOfSegment + 1));
  }

  indexSet.initDependencyGraph();

  for (int i = 0; i < numSegments; ++i) {
    RAJA::DepGraphNode* node = indexSet.getSegmentInfo(i)->getDepGraphNode();
    if (i < numSegments - 1) {    
      node->numDepTasks() = 1;
      node->depTaskNum(0) = i + 1;
    }
    if (i > 0) {
      node->semaphoreValue() = 1;
    }
  }
  indexSet.finalizeDependencyGraph();


  std::set<int> visitedNodes;
  RAJA::forall<ExecPolicy>(indexSet,
    [&](int i) {
    for (int j = 0; j < i; ++j) {
      EXPECT_TRUE(visitedNodes.find(j) != visitedNodes.end()) << "On task: " << i << ", unable to find dependent node: " << j << "\n";
    }
    visitedNodes.insert(i);
  });

}

TYPED_TEST_P(DependentIndexSetTest, BackwardDependentIndexSets)
{
  using ExecPolicy = TypeParam;

  const int numSegments = 10;
  const int sizeOfSegment = 1;
  RAJA::IndexSet indexSet;
  
  for (int i = 0; i < numSegments; ++i) {
    indexSet.push_back(RAJA::RangeSegment(i * sizeOfSegment, i * sizeOfSegment + 1));
  }

  indexSet.initDependencyGraph();

  for (int i = 0; i < numSegments; ++i) {
    RAJA::DepGraphNode* node = indexSet.getSegmentInfo(i)->getDepGraphNode();
    if (i > 0) {    
      node->numDepTasks() = 1;
      node->depTaskNum(0) = i - 1;
    }
    if (i < numSegments - 1) {
      node->semaphoreValue() = 1;
    }
  }
  indexSet.finalizeDependencyGraph();


  std::set<int> visitedNodes;
  RAJA::forall<ExecPolicy>(indexSet,
    [&](int i) {
    for (int j = numSegments - 1; j > i; --j) {
      EXPECT_TRUE(visitedNodes.find(j) != visitedNodes.end()) << "On task: " << i << ", unable to find dependent node: " << j << "\n";
    }
    visitedNodes.insert(i);
  });

}

REGISTER_TYPED_TEST_CASE_P(DependentIndexSetTest, ForwardDependentIndexSets, BackwardDependentIndexSets);

#ifdef RAJA_ENABLE_AGENCY

using agencyExecTypes = ::testing::Types<
  RAJA::IndexSet::ExecPolicy<RAJA::agency_taskgraph_parallel_segit, RAJA::agency_sequential_exec>,
  RAJA::IndexSet::ExecPolicy<RAJA::agency_taskgraph_parallel_segit, RAJA::seq_exec> >;

INSTANTIATE_TYPED_TEST_CASE_P(AgencyTests, DependentIndexSetTest, agencyExecTypes);

#ifdef RAJA_ENABLE_OPENMP

using agencyOMPExecTypes = ::testing::Types<
  RAJA::IndexSet::ExecPolicy<RAJA::agency_taskgraph_omp_segit, RAJA::agency_sequential_exec>,
  RAJA::IndexSet::ExecPolicy<RAJA::agency_taskgraph_omp_segit, RAJA::seq_exec> >;

INSTANTIATE_TYPED_TEST_CASE_P(AgencyOMPTests, DependentIndexSetTest, agencyOMPExecTypes);
#endif
#endif

#ifdef RAJA_ENABLE_OPENMP

using ompExecTypes = ::testing::Types<
  RAJA::IndexSet::ExecPolicy<RAJA::omp_taskgraph_segit, RAJA::seq_exec> >;

INSTANTIATE_TYPED_TEST_CASE_P(OMPTests, DependentIndexSetTest, ompExecTypes);

#endif


#if !defined(RAJA_COMPILER_XLC12) && 1
TEST_F(IndexSetTest, conditionalOperation_even_indices)
{

  RAJA::RAJAVec<RAJA::Index_type> even_indices;
  getIndicesConditional(even_indices, index_sets_[0], [](RAJA::Index_type idx) {
      return !(idx % 2);
  });

  RAJA::RAJAVec<RAJA::Index_type> ref_even_indices;
  for (size_t i = 0; i < is_indices.size(); ++i) {
      RAJA::Index_type idx = is_indices[i];
      if (idx % 2 == 0) {
          ref_even_indices.push_back(idx);
      }
  }

  EXPECT_EQ(even_indices.size(), ref_even_indices.size());
  for (size_t i = 0; i < ref_even_indices.size(); ++i) {
      EXPECT_EQ(even_indices[i], ref_even_indices[i]);
  }
}

TEST_F(IndexSetTest, conditionalOperation_lt300_indices)
{
  RAJA::RAJAVec<RAJA::Index_type> lt300_indices;
  getIndicesConditional(lt300_indices, index_sets_[0], [](RAJA::Index_type idx) {
      return (idx < 300);
  });

  RAJA::RAJAVec<RAJA::Index_type> ref_lt300_indices;
  for (size_t i = 0; i < is_indices.size(); ++i) {
      RAJA::Index_type idx = is_indices[i];
      if (idx < 300) {
          ref_lt300_indices.push_back(idx);
      }
  }

  EXPECT_EQ(lt300_indices.size(), ref_lt300_indices.size());
  for (size_t i = 0; i < ref_lt300_indices.size(); ++i) {
      EXPECT_EQ(lt300_indices[i], ref_lt300_indices[i]);
  }
}
#endif // !defined(RAJA_COMPILER_XLC12) && 1
