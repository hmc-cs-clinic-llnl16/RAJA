/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see RAJA/LICENSE.
 */

//
// Header file defining methods that build index sets in various ways
// for testing...
//

#ifndef RAJA_test_unit_cpu_buildIndexSet_HXX
#define RAJA_test_unit_cpu_buildIndexSet_HXX

#include "RAJA/RAJA.hxx"
#include <vector>

//
// Enum for different hybrid initialization procedures.
//
enum IndexSetBuildMethod {
  AddSegments = 0,
  AddSegmentsReverse,
  AddSegmentsNoCopy,
  AddSegmentsNoCopyReverse,
  MakeViewRange,
  MakeViewArray,
#if defined(RAJA_USE_STL)
  MakeViewVector,
#endif

  NumBuildMethods
};

//
//  Initialize index set by adding segments as indicated by enum value.
//  Return last index in IndexSet.
//
RAJA::Index_type buildIndexSet(RAJA::IndexSet* hindex, IndexSetBuildMethod build_method)
{
  //
  // Record last index in index set for return.
  //
  RAJA::Index_type last_indx = 0;

  //
  // Build vector of integers for creating RAJA::ListSegments.
  //
  RAJA::Index_type lindx_end = 0;
  RAJA::RAJAVec<RAJA::Index_type> lindices;
  for (RAJA::Index_type i = 0; i < 5; ++i) {
    RAJA::Index_type istart = lindx_end;
    lindices.push_back(istart + 1);
    lindices.push_back(istart + 4);
    lindices.push_back(istart + 5);
    lindices.push_back(istart + 9);
    lindices.push_back(istart + 10);
    lindices.push_back(istart + 11);
    lindices.push_back(istart + 12);
    lindices.push_back(istart + 14);
    lindices.push_back(istart + 15);
    lindices.push_back(istart + 21);
    lindices.push_back(istart + 27);
    lindices.push_back(istart + 28);
    lindx_end = istart + 28;
  }

  //
  // Create a vector of interleaved Range and List segments.
  //

  const int seg_chunk_size = 5;
  RAJA::RAJAVec<RAJA::BaseSegment*> segments;
  for (int i = 0; i < seg_chunk_size; ++i) {
    RAJA::Index_type rbeg;
    RAJA::Index_type rend;
    RAJA::Index_type lseg_len = lindices.size();
    RAJA::RAJAVec<RAJA::Index_type> lseg(lseg_len);

    // Create Range segment
    rbeg = last_indx + 2;
    rend = rbeg + 32;
    segments.push_back(new RAJA::RangeSegment(rbeg, rend));
    last_indx = rend;

    // Create List segment
    for (RAJA::Index_type i = 0; i < lseg_len; ++i) {
      lseg[i] = lindices[i] + last_indx;
    }
    segments.push_back(new RAJA::ListSegment(&lseg[0], lseg_len));
    last_indx = lseg[lseg_len - 1];

    // Create Range segment
    rbeg = last_indx + 16;
    rend = rbeg + 128;
    segments.push_back(new RAJA::RangeSegment(rbeg, rend));
    last_indx = rend;

    // Create Range segment
    rbeg = last_indx + 4;
    rend = rbeg + 256;
    segments.push_back(new RAJA::RangeSegment(rbeg, rend));
    last_indx = rend;

    // Create List segment
    for (RAJA::Index_type i = 0; i < lseg_len; ++i) {
      lseg[i] = lindices[i] + last_indx + 5;
    }
    segments.push_back(new RAJA::ListSegment(&lseg[0], lseg_len));
    last_indx = lseg[lseg_len - 1];
  }

  //
  // Generate IndexSet from segments using specified build method.
  //
  switch (build_method) {
    case AddSegments: {
      for (size_t i = 0; i < segments.size(); ++i) {
        hindex[build_method].push_back(*segments[i]);
      }

      break;
    }

    case AddSegmentsReverse: {
      int last_i = static_cast<int>(segments.size() - 1);
      for (int i = last_i; i >= 0; --i) {
        hindex[build_method].push_front(*segments[i]);
      }

      break;
    }

    case AddSegmentsNoCopy: {
      RAJA::IndexSet& iset_master = hindex[0];

      for (size_t i = 0; i < iset_master.getNumSegments(); ++i) {
        hindex[build_method].push_back_nocopy(iset_master.getSegment(i));
      }

      break;
    }

    case AddSegmentsNoCopyReverse: {
      RAJA::IndexSet& iset_master = hindex[0];

      int last_i = static_cast<int>(iset_master.getNumSegments() - 1);
      for (int i = last_i; i >= 0; --i) {
        hindex[build_method].push_front_nocopy(iset_master.getSegment(i));
      }

      break;
    }

    case MakeViewRange: {
      RAJA::IndexSet& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();

      RAJA::IndexSet* iset_view = iset_master.createView(0, num_segs);

      for (size_t i = 0; i < iset_view->getNumSegments(); ++i) {
        hindex[build_method].push_back_nocopy(iset_view->getSegment(i));
      }

      break;
    }

    case MakeViewArray: {
      RAJA::IndexSet& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      int* segIds = new int[num_segs];
      for (size_t i = 0; i < num_segs; ++i) {
        segIds[i] = i;
      }

      RAJA::IndexSet* iset_view = iset_master.createView(segIds, num_segs);

      for (size_t i = 0; i < iset_view->getNumSegments(); ++i) {
        hindex[build_method].push_back_nocopy(iset_view->getSegment(i));
      }

      delete[] segIds;

      break;
    }

#if defined(RAJA_USE_STL)
    case MakeViewVector: {
      RAJA::IndexSet& iset_master = hindex[0];
      size_t num_segs = iset_master.getNumSegments();
      std::vector<int> segIds(num_segs);
      for (int i = 0; i < num_segs; ++i) {
        segIds[i] = i;
      }

      RAJA::IndexSet* iset_view = iset_master.createView(segIds);

      for (size_t i = 0; i < iset_view->getNumSegments(); ++i) {
        hindex[build_method].push_back_nocopy(iset_view->getSegment(i));
      }

      break;
    }
#endif

    default: {
    }

  }  // switch (build_method)

  for (size_t i = 0; i < segments.size(); ++i) {
    delete segments[i];
  }

  return last_indx;
}
#endif // endif header guard
