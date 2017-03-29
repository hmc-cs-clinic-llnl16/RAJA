/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing forallN Agency constructs.
 *
 ******************************************************************************
 */

#ifndef RAJA_forallN_agency_HXX__
#define RAJA_forallN_agency_HXX__

#include "RAJA/config.hxx"
#include <thread>

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
#include "RAJA/ThreadUtils_CPU.hxx"

#include "agency/agency.hpp"
#include "agency/experimental.hpp"

namespace RAJA
{

/******************************************************************
 *  ForallN Agency policies
 ******************************************************************/

template <typename Agent, typename Worker>
struct ForallN_Agency_Parallel_Tag { };

template <typename Agent, typename Worker, typename NEXT = Execute>
struct Agency_Parallel {
  using PolicyTag = ForallN_Agency_Parallel_Tag<Agent, Worker>;
  using NextPolicy = NEXT;
};

/******************************************************************
 *  ForallN policies
 ******************************************************************/

template <typename Agent, typename Worker, typename... PREST>
struct ForallN_Executor<ForallN_PolicyPair<agency_base<Agent, Worker>,
                                           RangeSegment>,
                        ForallN_PolicyPair<agency_base<Agent, Worker>,
                                           RangeSegment>,
                        PREST...> {
  using PolicyPairType = ForallN_PolicyPair<agency_base<Agent, Worker>, RangeSegment>;
  PolicyPairType iset_i, iset_j;

  using NextExec = ForallN_Executor<PREST...>;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(
      PolicyPairType const &iseti_,
      PolicyPairType const &isetj_,
      PREST const &... prest)
      : iset_i(iseti_), iset_j(isetj_), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    int begin_i = iset_i.getBegin();
    int begin_j = iset_j.getBegin();
    int end_i = iset_i.getEnd();
    int end_j = iset_j.getEnd();

    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);

    auto numThreads = getMaxReduceThreadsCPU();
    auto tiles = agency::experimental::tile_evenly(
        agency::experimental::interval(begin_i, end_i), numThreads);

    agency::bulk_invoke(Worker{}(tiles.size()),
                        [=](Agent& self) {
                            for (Index_type i : tiles[self.index]) {
                                for (Index_type j = begin_j; j < end_j; ++j) {
                                    outer(i, j);
                                }
                            }
                        });
    }
};

template <typename Agent, typename  Worker, typename... PREST>
struct ForallN_Executor<ForallN_PolicyPair<agency_base<Agent, Worker>,
                                           RangeSegment>,
                        ForallN_PolicyPair<agency_base<Agent, Worker>,
                                           RangeSegment>,
                        ForallN_PolicyPair<agency_base<Agent, Worker>,
                                           RangeSegment>,
                        PREST...> {
  using PolicyPairType = ForallN_PolicyPair<agency_base<Agent, Worker>, RangeSegment>;
  PolicyPairType iset_i, iset_j, iset_k;

  using NextExec = ForallN_Executor<PREST...>;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(
      PolicyPairType const &iseti_,
      PolicyPairType const &isetj_,
      PolicyPairType const &isetk_,
      PREST const &... prest)
      : iset_i(iseti_), iset_j(isetj_), iset_k(isetk_), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    int begin_i = iset_i.getBegin();
    int begin_j = iset_j.getBegin();
    int begin_k = iset_k.getBegin();
    int end_i = iset_i.getEnd();
    int end_j = iset_j.getEnd();
    int end_k = iset_k.getEnd();

    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);

    auto numThreads = getMaxReduceThreadsCPU();
    auto tiles = agency::experimental::tile_evenly(
        agency::experimental::interval(begin_i, end_i), numThreads);

    agency::bulk_invoke(Worker{}(tiles.size()),
                        [=](Agent& self) {
                            for (Index_type i : tiles[self.index]) {
                                for (Index_type j = begin_j; j < end_j; ++j) {
                                    for (Index_type k = begin_k; k < end_k; ++k) {
                                        outer(i, j, k);
                                    }
                                }
                            }
                        });
    }
};

/******************************************************************
 *  forallN_policy()
 ******************************************************************/


/*!
 * \brief Tiling policy front-end function.
 */
template <typename POLICY, typename BODY, typename Agent, typename Worker, typename... PARGS>
RAJA_INLINE void forallN_policy(ForallN_Agency_Parallel_Tag<Agent, Worker>,
                                BODY body,
                                PARGS... pargs)
{
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;

  // It works if we leave it like this, but not if the part below
  // is commented out.
  // This feels weird...
  forallN_policy<NextPolicy>(NextPolicyTag(), body, pargs...);

 //  auto numThreads = getMaxReduceThreadsCPU();

 //  agency::bulk_invoke(Worker{}(numThreads),
 //                      [=](Agent&) {
 //                          forallN_policy<NextPolicy>(NextPolicyTag(), body, pargs...);
 //                      });
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_AGENCY)

#endif  // closing endif for header file include guard
