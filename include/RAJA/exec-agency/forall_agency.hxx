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

#include "agency/agency.hpp"

#include <thread>
#include <iterator>

namespace RAJA
{

template <typename Func, typename Agent, typename Worker>
RAJA_INLINE void forall(const agency_base<Agent, Worker>&, 
                        const rangeSegment& iter, 
                        Func&& loop_body)
{
  auto numThreads = max(std::thread::hardware_concurrency(), 1);
  auto workPerThread = std::distance(iter.getBegin(), iter.getEnd()) / numThreads;

  agency::bulk_invoke(Worker(numThreads),
                      [=](Agent& self) {
                        auto start = workPerThread * self.index();
                        auto end = self.index() == (numThreads - 1) 
                                     ? iter.getEnd()
                                     : start + workPerThread;
                        for (auto i = start; i < end; ++i) {
                          loop_body(i);  
                        }
                      });
}

template <typename Iterable, typename Func, typename Agent, typename Worker>
RAJA_INLINE void forall(const agency_base<Agent, Worker>&,
                        Iterable&& iter,
                        Func&& loop_body)
{
  auto begin = std::begin(iter);

  auto numThreads = max(std::thread::hardware_concurrency(), 1);
  auto workPerThread = std::distance(begin, std::end(iter)) / numThreads;

  agency::bulk_invoke(Worker(numThreads),
                      [=](Agent& self) {
                        auto start = workPerThread * self.index();
                        auto end = self.index() == (numThreads - 1) 
                                     ? iter.getEnd()
                                     : start + workPerThread;
                        for (auto i = start; i < end; ++i) {
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
  auto begin = std::begin(iter);

  auto numThreads = max(std::thread::hardware_concurrency(), 1);
  auto workPerThread = std::distance(begin, std::end(iter)) / numThreads;

  agency::bulk_invoke(Worker(numThreads),
                      [=](Agent& self) {
                        auto start = workPerThread * self.index();
                        auto end = self.index() == (numThreads - 1) 
                                     ? iter.getEnd()
                                     : start + workPerThread;
                        for (auto i = start; i < end; ++i) {
                          loop_body(i + icount, begin[i]);  
                        }
                      });
}

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_AGENCY)

#endif  // closing endif for header file include guard
