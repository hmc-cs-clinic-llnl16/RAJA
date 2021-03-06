/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for Agency execution.
 *
 *          These methods work only on platforms that support Agency.
 *
 ******************************************************************************
 */

#ifndef RAJA_agency_HXX
#define RAJA_agency_HXX

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

#include "agency/agency.hpp"
#include "agency/experimental.hpp"

#if defined(RAJA_ENABLE_OPENMP)
#   include "agency/omp.hpp"
#endif

#if defined(RAJA_ENABLE_CUDA)
#   include "agency/cuda.hpp"
#endif

namespace RAJA {
//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

template <typename AGENT, typename WORKER>
struct agency_base { 
    using Agent_t = AGENT;
    using Worker_t = WORKER;
};

using agency_parallel_exec = agency_base<
  agency::parallel_agent, 
  decltype(agency::par)
>;

using agency_sequential_exec = agency_base<
  agency::sequenced_agent, 
  decltype(agency::seq)
>;

#if defined(RAJA_ENABLE_OPENMP)
using agency_omp_parallel_exec = agency_base<
  agency::parallel_agent, 
  decltype(agency::omp::par)
>;
#endif

#if defined(RAJA_ENABLE_CUDA)
// Wrapper functor because we can't template on the overloaded function grid
struct AgencyCudaGrid {
  auto operator()(size_t num_blocks, size_t num_threads) ->
      decltype(agency::cuda::grid(num_blocks, num_threads))
  {
     return agency::cuda::grid(num_blocks, num_threads); 
  }
};

template <typename AGENT, typename WORKER, size_t BLOCK_SIZE, bool Async>
struct agency_cuda_base {
    using Agent_t = AGENT;
    using Worker_t = WORKER;
    static const size_t blockSize = BLOCK_SIZE;
    static const bool async = Async;
};

template <size_t BLOCK_SIZE, bool Async = false>
using agency_cuda_exec = agency_cuda_base<
  agency::cuda::grid_agent,
  AgencyCudaGrid,
  BLOCK_SIZE,
  Async
>;
#endif

//
//////////////////////////////////////////////////////////////////////
//
// Taskgraph policies
//
//////////////////////////////////////////////////////////////////////
//
template <typename AGENT, typename WORKER>
struct agency_taskgraph_base { };

using agency_taskgraph_parallel_segit = agency_taskgraph_base<
  agency::parallel_agent,
  decltype(agency::par)
>;

#if defined(RAJA_ENABLE_OPENMP)
using agency_taskgraph_omp_segit = agency_taskgraph_base<
  agency::parallel_agent,
  decltype(agency::omp::par)
>;
#endif

//
//////////////////////////////////////////////////////////////////////
//
// Reduction policies
//
//////////////////////////////////////////////////////////////////////
//

struct agency_reduce { };

}  // closing brace for RAJA namespace

#include "RAJA/exec-agency/forall_agency.hxx"
#include "RAJA/exec-agency/reduce_agency.hxx"
// TODO: Implement scan
// #include "RAJA/exec-agency/scan_agency.hxx"
 
#if defined(RAJA_ENABLE_NESTED)
#    include "RAJA/exec-agency/forallN_agency.hxx"
#endif

#if defined(RAJA_ENABLE_CUDA)
#    include "RAJA/exec-agency/forall_cuda_agency.hxx"
#endif

#endif  // closing endif for if defined(RAJA_ENABLE_AGENCY)

#endif  // closing endif for header file include guard
