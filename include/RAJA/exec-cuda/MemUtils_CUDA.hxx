/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_CUDA_HXX
#define RAJA_MemUtils_CUDA_HXX

#include "RAJA/config.hxx"

#if defined(RAJA_ENABLE_CUDA)

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

#include "RAJA/int_datatypes.hxx"

namespace RAJA
{
/// Maximum number of blocks that RAJA will launch
/// threads are reused to finish all work
#define RAJA_CUDA_MAX_NUM_BLOCKS (1024 * 16)

/// Size of reduction memory block for each reducer object (value based on
/// rough estimate of "worst case" -- need to think more about this...
#define RAJA_CUDA_REDUCE_BLOCK_LENGTH RAJA_CUDA_MAX_NUM_BLOCKS

/// Reduction Tallies are computed into a small block to minimize UM migration
#define RAJA_CUDA_REDUCE_TALLY_LENGTH RAJA_MAX_REDUCE_VARS

/// Should be large enough for all types for which cuda atomics exist
/// includes the size of the index variable for Loc reductions
#define RAJA_CUDA_REDUCE_VAR_MAXSIZE 16

#define STR(x) #x
#define MACROSTR(x) STR(x)

#define RAJA_STRUCT_ALIGNAS alignas(DATA_ALIGN)


/// dummy types for use in allocating arrays and distributing array segments
struct CudaReductionDummyDataType {
	unsigned char data[RAJA_CUDA_REDUCE_VAR_MAXSIZE];
};

struct RAJA_STRUCT_ALIGNAS CudaReductionDummyBlockType {
	CudaReductionDummyDataType values[RAJA_CUDA_REDUCE_BLOCK_LENGTH];
};

struct CudaReductionDummyTallyType {
	CudaReductionDummyDataType dummy_val;
	CudaReductionDummyDataType dummy_idx;
};

typedef unsigned int GridSizeType;

/// types used to simplify typed memory use in reductions
/// these types fit within the dummy types, checked in static asserts in reduction classes
///
/// Each ReduceSum, ReduceMinLoc, or ReduceMaxLoc object uses retiredBlocks
/// as a way to complete the reduction in a single pass. Although the algorithm
/// updates retiredBlocks via an atomicAdd(int) the actual reduction values
/// do not use atomics and require a finishing stage performed
/// by the last block.
///
template<typename T>
struct RAJA_STRUCT_ALIGNAS CudaReductionBlockType {
	T values[RAJA_CUDA_REDUCE_BLOCK_LENGTH];
};

template<typename T>
struct CudaReductionLocType {
	T val;
	Index_type idx;
};

template<typename T>
struct RAJA_STRUCT_ALIGNAS CudaReductionLocBlockType {
	T values[RAJA_CUDA_REDUCE_BLOCK_LENGTH];
	Index_type indices[RAJA_CUDA_REDUCE_BLOCK_LENGTH];
};

template<typename T>
struct CudaReductionTallyType {
	T tally;
	GridSizeType maxGridSize;
	GridSizeType retiredBlocks;
};

template<typename T>
struct CudaReductionTallyTypeAtomic {
	T tally;
};

template<typename T>
struct CudaReductionLocTallyType {
	CudaReductionLocType<T> tally;
	GridSizeType maxGridSize;
	GridSizeType retiredBlocks;
};


/*!
*************************************************************************
*
* Return next available valid reduction id, or complain and exit if
* no valid id is available.
*
*************************************************************************
*/
int getCudaReductionId();

/*!
*************************************************************************
*
* Release given redution id and make inactive.
*
*************************************************************************
*/
void releaseCudaReductionId(int id);

void getCudaReductionTallyBlock(int id, void** host_tally, void** device_tally);

void releaseCudaReductionTallyBlock(int id);

void beforeCudaKernelLaunch();

void afterCudaKernelLaunch();

void beforeCudaReadTallyBlockAsync(int id);

void beforeCudaReadTallyBlock(int id);

/*!
 ******************************************************************************
 *
 * \brief  Earmark amount of device shared memory and get byte offset into 
 *         device shared memory if in a RAJA forall.
 *
 ******************************************************************************
 */
int getCudaSharedmemOffset(int id, int amount);

/*!
 ******************************************************************************
 *
 * \brief  Get the amount in bytes of shared memory required for thie current
 *         kernel launch.
 *
 ******************************************************************************
 */
int getCudaSharedmemAmount();

/*!
 ******************************************************************************
 *
 * \brief  Free managed memory block used in RAJA-Cuda reductions.
 *
 ******************************************************************************
 */

void freeCudaReductionTallyBlock();

/*!
 ******************************************************************************
 *
 * \brief  Return pointers into shared memory blocks for RAJA-CUDA reduction
 *         with given id.
 *
 *         Allocates data block if it isn't allocated already.
 *
 * NOTE: Block size will be:
 *
 *          sizeof(CudaReductionBlockDataType) *
 *            RAJA_MAX_REDUCE_VARS * ( RAJA_CUDA_REDUCE_BLOCK_LENGTH + 1 + 1 )
 *
 *       For each reducer object, we want a chunk of managed memory that
 *       holds RAJA_CUDA_REDUCE_BLOCK_LENGTH slots for the reduction
 *       value for each thread, a single slot for the global reduced value
 *       across grid blocks, and a single slot for the max grid size
 *
 ******************************************************************************
 */
 
void getCudaReductionMemBlock(int id, void** device_memblock);

/*!
 ******************************************************************************
 *
 * \brief  Free managed memory blocks used in RAJA-Cuda reductions.
 *
 ******************************************************************************
 */
void freeCudaReductionMemBlock();

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
