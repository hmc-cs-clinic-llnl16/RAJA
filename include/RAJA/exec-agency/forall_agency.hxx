/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for Agency.
 *
 *          These methods should work on any platform that supports Agency.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_agency_HXX
#define RAJA_forall_agency_HXX

#include "RAJA/config.hxx"

#if defined(RAJA_ENABLE_AGENCY)

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
// For additional details, please also read RAJA/LICENSE.
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

#include "RAJA/int_datatypes.hxx"
#include "RAJA/fault_tolerance.hxx"
#include "RAJA/segment_exec.hxx"
#include "RAJA/ThreadUtils_CPU.hxx"

#include "agency/agency.hpp"
#include "agency/experimental.hpp"

#include <iostream>
#include <thread>

namespace RAJA
{
  /// ASSUMPTIONS:
  ///   1- That we can get the number of threads this way
  ///   2- That the number of workers can be sensibly computed this way
  ///   3- That the number of workers computed this way works for 
  ///      sequenctuial execution.
  template <typename Func, typename Agent, typename Worker>
  RAJA_INLINE void forall(const agency_base<Agent, Worker>&, 
                          const RangeSegment& iter, 
                          Func&& loop_body)
  {
    auto numThreads = getMaxReduceThreadsCPU();
    auto tiles = agency::experimental::tile_evenly(
        agency::experimental::interval(iter.getBegin(), iter.getEnd()), numThreads);

    agency::bulk_invoke(Worker{}(tiles.size()),
                        [=](Agent& self) {
                            for (Index_type i : tiles[self.index()]) {
                                loop_body(i);
                            }
                        });
  }

  template <typename Iterable, typename Func, typename Agent, typename Worker>
  RAJA_INLINE void forall(const agency_base<Agent, Worker>&,
                          Iterable&& iter,
                          Func&& loop_body)
  {
    auto numThreads = getMaxReduceThreadsCPU();
    auto distance = std::end(iter) - std::begin(iter);

    auto tiles = agency::experimental::tile_evenly(
        agency::experimental::interval(0, distance), numThreads);

    agency::bulk_invoke(Worker{}(tiles.size()),
                        [=](Agent& self) {
                            auto begin= std::begin(iter);
                            for (Index_type i : tiles[self.index()]) {
                                loop_body(begin[i]);
                            }
                        });
  }

  template <typename Iterable, typename Func, typename Agent, typename Worker>
  RAJA_INLINE void forall_Icount(const agency_base<Agent, Worker>&,
      Iterable&& iter,
      Index_type icount,
      Func&& loop_body)
  {
    auto distance = std::distance(std::begin(iter), std::end(iter));
    auto numThreads = getMaxReduceThreadsCPU();

    auto tiles = agency::experimental::tile_evenly(
        agency::experimental::interval(0, distance), numThreads);

    agency::bulk_invoke(Worker{}(tiles.size()),
                        [=](Agent& self) {
                            auto begin= std::begin(iter);
                            for (Index_type i : tiles[self.index()]) {
                                loop_body(i + icount, begin[i]);
                            }
                        });
  }

template <typename SEG_EXEC_POLICY, typename LOOP_BODY, typename Worker>
RAJA_INLINE void forall(
    IndexSet::ExecPolicy<agency_taskgraph_base<agency::parallel_agent, Worker>, SEG_EXEC_POLICY>,
    const IndexSet& iset,
    LOOP_BODY loop_body)
{
  if (!iset.dependencyGraphSet()) {
    std::cerr << "\n RAJA IndexSet dependency graph not set "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    exit(1);
  }

  // Cast away const because we want it in the other overloaded versions,
  // but need to modify the index sets due to how dependent nodes are
  // implemented.
  IndexSet* ncis = const_cast<IndexSet*>(&iset);
  int numSegments = ncis->getNumSegments();
  
  agency::bulk_invoke(Worker{}(numSegments) , [=] (agency::parallel_agent& self) {
    int isi = self.index();
    IndexSetSegInfo* segInfo = const_cast<IndexSetSegInfo*>(ncis->getSegmentInfo(isi));
    DepGraphNode* task = const_cast<DepGraphNode*>(segInfo->getDepGraphNode());
    task->wait();

    executeRangeList_forall<SEG_EXEC_POLICY>(segInfo, loop_body);

    task->reset();

    

    if (task->numDepTasks() != 0) {
      for (int ii = 0; ii < task->numDepTasks(); ++ii) {
        // Alternateively, we could get the return value of this call
        // and actively launch the task if we are the last depedent
        // task. In that case, we would not need the semaphore spin
        // loop above.
        int seg = task->depTaskNum(ii);
        DepGraphNode* dep = const_cast<DepGraphNode*>(ncis->getSegmentInfo(seg)->getDepGraphNode());
        dep->satisfyOne();
      }
    }
  });
}


}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_AGENCY)

#endif  // closing endif for header file include guard
