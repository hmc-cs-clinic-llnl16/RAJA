/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for CUDA execution.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
 *
 ******************************************************************************
 */

#ifndef RAJA_reduce_cuda_HXX
#define RAJA_reduce_cuda_HXX

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

#include <cassert>

#include "RAJA/int_datatypes.hxx"

#include "RAJA/reducers.hxx"

#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"

#include "RAJA/exec-cuda/raja_cudaerrchk.hxx"


namespace RAJA
{


/*!
 ******************************************************************************
 *
 * \brief Method to shuffle 32b registers in sum reduction.
 *
 ******************************************************************************
 */
__device__ __forceinline__ double shfl_xor(double var, int laneMask)
{
  int lo = __shfl_xor(__double2loint(var), laneMask);
  int hi = __shfl_xor(__double2hiint(var), laneMask);
  return __hiloint2double(hi, lo);
}

// The following atomic functions need to be outside of the RAJA namespace
#include <cuda.h>

//
// Three different variants of min/max reductions can be run by choosing
// one of these macros. Only one should be defined!!!
//
//#define RAJA_USE_ATOMIC_ONE
#define RAJA_USE_ATOMIC_TWO
//#define RAJA_USE_NO_ATOMICS

#if 0
#define ull_to_double(x) __longlong_as_double(reinterpret_cast<long long>(x))

#define double_to_ull(x) \
  reinterpret_cast<unsigned long long>(__double_as_longlong(x))
#else
#define ull_to_double(x) __longlong_as_double(x)

#define double_to_ull(x) __double_as_longlong(x)
#endif

template<typename T>
__device__ inline void _atomicMin(T* address, T value)
{
  atomicMin(address, value);
}

template<typename T>
__device__ inline void _atomicMax(T* address, T value)
{
  atomicMax(address, value);
}

template<typename T>
__device__ inline void _atomicAdd(T* address, T value)
{
  atomicAdd(address, value);
}

#if defined(RAJA_USE_ATOMIC_ONE)
/*!
 ******************************************************************************
 *
 * \brief Atomic min and max update methods used to gather current
 *        min or max reduction value.
 *
 ******************************************************************************
 */
template<>
__device__ inline void _atomicMin(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp > value) {
    unsigned long long oldval, newval, readback;
    oldval = double_to_ull(temp);
    newval = double_to_ull(value);
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    while ((readback = atomicCAS(address_as_ull, oldval, newval)) != oldval) {
      oldval = readback;
      newval = double_to_ull(RAJA_MIN(ull_to_double(oldval), value));
    }
  }
}
///
 template<>
__device__ inline void _atomicMin(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp > value) {
    int oldval, newval, readback;
    oldval = __float_as_int(temp);
    newval = __float_as_int(value);
    int *address_as_i = reinterpret_cast<int *>(address);

    while ((readback = atomicCAS(address_as_i, oldval, newval)) != oldval) {
      oldval = readback;
      newval = __float_as_int(RAJA_MIN(__int_as_float(oldval), value));
    }
  }
}
///
template<>
__device__ inline void _atomicMax(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp < value) {
    unsigned long long oldval, newval, readback;
    oldval = double_to_ull(temp);
    newval = double_to_ull(value);
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    while ((readback = atomicCAS(address_as_ull, oldval, newval)) != oldval) {
      oldval = readback;
      newval = double_to_ull(RAJA_MAX(ull_to_double(oldval), value));
    }
  }
}
///
template<>
__device__ inline void _atomicMax(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp < value) {
    int oldval, newval, readback;
    oldval = __float_as_int(temp);
    newval = __float_as_int(value);
    int *address_as_i = reinterpret_cast<int *>(address);

    while ((readback = atomicCAS(address_as_i, oldval, newval)) != oldval) {
      oldval = readback;
      newval = __float_as_int(RAJA_MAX(__int_as_float(oldval), value));
    }
  }
}
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 350
/// don't specialize for 64-bit min/max if they exist
#else
/// implement 64-bit min/max if they don't exist
template<>
__device__ inline void _atomicMin(unsigned long long int *address, unsigned long long int value)
{
  unsigned long long int temp = *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp > value) {
    unsigned long long int oldval, newval;
    oldval = temp;
    newval = value;

    while ((readback = atomicCAS(address, oldval, newval)) != oldval) {
      oldval = readback;
      newval = RAJA_MIN(oldval, value);
    }
  }
  return readback;
}
///
template<>
__device__ inline void _atomicMax(unsigned long long int *address, unsigned long long int value)
{
  unsigned long long int readback = *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (readback < value) {
    unsigned long long int oldval, newval;
    oldval = readback;
    newval = value;

    while ((readback = atomicCAS(address, oldval, newval)) != oldval) {
      oldval = readback;
      newval = RAJA_MAX(oldval, value);
    }
  }
  return readback;
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
/// don't specialize for 64-bit add if it exists
#else
///
template<>
__device__ inline void _atomicAdd(double *address, double value)
{
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback = atomicCAS((unsigned long long *)address, oldval, newval))
         != oldval) {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  }
}
#endif

#elif defined(RAJA_USE_ATOMIC_TWO)

/*!
 ******************************************************************************
 *
 * \brief Alternative atomic min and max update methods used to gather current
 *        min or max reduction value.
 *
 *        These appear to be more robust than the ones above, not sure why.
 *
 ******************************************************************************
 */
template<>
__device__ inline void _atomicMin(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp > value) {
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed;
    unsigned long long oldval = double_to_ull(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_ull,
                    assumed,
                    double_to_ull(RAJA_MIN(ull_to_double(assumed), value)));
    } while (assumed != oldval);
  }
}
///
template<>
__device__ inline void _atomicMin(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp > value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_i,
                    assumed,
                    __float_as_int(RAJA_MIN(__int_as_float(assumed), value)));
    } while (assumed != oldval);
  }
}
///
template<>
__device__ inline void _atomicMax(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp < value) {
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed;
    unsigned long long oldval = double_to_ull(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_ull,
                    assumed,
                    double_to_ull(RAJA_MAX(ull_to_double(assumed), value)));
    } while (assumed != oldval);
  }
}
///
template<>
__device__ inline void _atomicMax(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp < value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_i,
                    assumed,
                    __float_as_int(RAJA_MAX(__int_as_float(assumed), value)));
    } while (assumed != oldval);
  }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 350
/// don't specialize for 64-bit min/max if they exist
#else
///
template<>
__device__ inline void _atomicMin(unsigned long long int *address, unsigned long long int value)
{
  unsigned long long int temp = *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp > value) {
    unsigned long long int assumed;
    unsigned long long int oldval = temp;
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address,
                    assumed,
                    RAJA_MIN(assumed, value));
    } while (assumed != oldval);
  }
}
///
template<>
__device__ inline void _atomicMax(unsigned long long int *address, unsigned long long int value)
{
  unsigned long long int temp = *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp < value) {
    unsigned long long int assumed;
    unsigned long long int oldval = temp;
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address,
                    assumed,
                    RAJA_MAX(assumed, value));
    } while (assumed != oldval);
  }
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
/// don't specialize for doubles if they exist
#else
///
template<>
__device__ inline void _atomicAdd(double *address, double value)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int oldval = *address_as_ull, assumed;

  do {
    assumed = oldval;
    oldval =
        atomicCAS(address_as_ull,
                  assumed,
                  __double_as_longlong(__longlong_as_double(oldval) + value));
  } while (assumed != oldval);
}
#endif

#elif defined(RAJA_USE_NO_ATOMICS)

// Noting to do here...

#else

#error one of the options for using/not using atomics must be specified

#endif

/*!
 ******************************************************************************
 *
 * \brief Kernel to set memory at to 0 before offset and then val in
 *        next N locations after offset
 *
 ******************************************************************************
 */
template <typename T0>
__global__
void rajaCudaMemsetType(T0* ptr0, T0 val0, int N0)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N0) {
        ptr0[i] = val0;
    }
}

template <typename T0, typename T1>
__global__
void rajaCudaMemsetType(T0* ptr0, T0 val0, int N0,
                        T1* ptr1, T1 val1, int N1)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N0) {
      ptr0[i] = val0;
    } else if (i < N0+N1) {
      ptr1[i-N0] = val1;
    }
}

template <typename T0, typename T1, typename T2>
__global__
void rajaCudaMemsetType(T0* ptr0, T0 val0, int N0,
                        T1* ptr1, T1 val1, int N1,
                        T2* ptr2, T2 val2, int N2)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N0) {
      ptr0[i] = val0;
    } 
    else if (i < N0+N1) {
      ptr1[i-N0] = val1;
    }
    else if (i < N0+N1+N2) {
      ptr2[i-(N0+N1)] = val2;
    }
}

template <typename T0, typename T1, typename T2, typename T3>
__global__
void rajaCudaMemsetType(T0* ptr0, T0 val0, int N0,
                        T1* ptr1, T1 val1, int N1,
                        T2* ptr2, T2 val2, int N2,
                        T3* ptr3, T3 val3, int N3)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N0) {
      ptr0[i] = val0;
    } 
    else if (i < N0+N1) {
      ptr1[i-N0] = val1;
    }
    else if (i < N0+N1+N2) {
      ptr2[i-(N0+N1)] = val2;
    }
    else if (i < N0+N1+N2+N3) {
      ptr3[i-(N0+N1+N2)] = val3;
    }
}


//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Min reduction class template for use in CUDA kernels.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceMin<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceMin(T init_val)
  {
    m_is_copy = false;
    m_reduced_val = init_val;
    m_myID = getCudaReductionId();
    m_tallydata = static_cast<CudaReductionTallyType<T>*>(getCudaReductionTallyBlock(m_myID));

    rajaCudaMemsetType<T, T>
      <<<2, 1>>>
      ( &m_tallydata->tally, init_val, 1,
        &m_tallydata->initVal, init_val, 1 );
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceMin(const ReduceMin<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceMin<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());
    m_reduced_val = m_tallydata->tally;
    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // Method that updates min value in proper device memory block locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceMin<cuda_reduce<BLOCK_SIZE>, T> min(T val) const
  {
    __shared__ T sd[BLOCK_SIZE];

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i] = m_reduced_val;
      }
    }
    __syncthreads();

    sd[threadId] = val;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + i]);
      }
      __syncthreads();
    }

    if (threadId < 16) {
      sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + 16]);
    }
    __syncthreads();

    if (threadId < 8) {
      sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + 8]);
    }
    __syncthreads();

    if (threadId < 4) {
      sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + 4]);
    }
    __syncthreads();

    if (threadId < 2) {
      sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + 2]);
    }
    __syncthreads();

    if (threadId < 1) {
      sd[0] = RAJA_MIN(sd[0], sd[1]);
      _atomicMin<T>(&(m_tallydata->tally), sd[0]);
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMin<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;

  T m_reduced_val;

  CudaReductionTallyType<T> *m_tallydata;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck = 
      (  (sizeof(T) <= sizeof(CudaReductionDummyDataType))
      && (sizeof(CudaReductionTallyType<T>) <= sizeof(CudaReductionDummyTallyType))
      && (sizeof(CudaReductionBlockType<T>) <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck, "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

/*!
 ******************************************************************************
 *
 * \brief  Max reduction class template for use in CUDA kernels.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceMax<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceMax(T init_val)
  {
    m_is_copy = false;
    m_reduced_val = init_val;
    m_myID = getCudaReductionId();
    m_tallydata = static_cast<CudaReductionTallyType<T>*>(getCudaReductionTallyBlock(m_myID));

    rajaCudaMemsetType<T, T>
      <<<2, 1>>>
      ( &m_tallydata->tally, init_val, 1,
        &m_tallydata->initVal, init_val, 1 );
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceMax(const ReduceMax<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceMax<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced max value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());
    m_reduced_val = m_tallydata->tally;
    return m_reduced_val;
  }

  //
  // Method that returns reduced max value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // Method that updates max value in proper device memory block locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceMax<cuda_reduce<BLOCK_SIZE>, T> max(T val) const
  {
    __shared__ T sd[BLOCK_SIZE];

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i] = m_reduced_val;
      }
    }
    __syncthreads();

    sd[threadId] = val;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + i]);
      }
      __syncthreads();
    }

    if (threadId < 16) {
      sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + 16]);
    }
    __syncthreads();

    if (threadId < 8) {
      sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + 8]);
    }
    __syncthreads();

    if (threadId < 4) {
      sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + 4]);
    }
    __syncthreads();

    if (threadId < 2) {
      sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + 2]);
    }
    __syncthreads();

    if (threadId < 1) {
      sd[0] = RAJA_MAX(sd[0], sd[1]);
      _atomicMax<T>(&(m_tallydata->tally), sd[0]);
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMax<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;

  T m_reduced_val;

  CudaReductionTallyType<T> *m_tallydata;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck = 
      (  (sizeof(T) <= sizeof(CudaReductionDummyDataType))
      && (sizeof(CudaReductionTallyType<T>) <= sizeof(CudaReductionDummyTallyType))
      && (sizeof(CudaReductionBlockType<T>) <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck, "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

/*!
 ******************************************************************************
 *
 * \brief  Sum reduction class template for use in CUDA kernel.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceSum<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceSum(T init_val)
  {
    m_is_copy = false;
    m_init_val = init_val;
    m_reduced_val = static_cast<T>(0);
    m_myID = getCudaReductionId();
    m_blockdata = static_cast<CudaReductionBlockType<T>*>(getCudaReductionMemBlock(m_myID));

    // Entire global shared memory block must be initialized to zero so
    // sum reduction is correct.
    rajaCudaMemsetType<GridSizeType, T>
      <<<((1+RAJA_CUDA_REDUCE_BLOCK_LENGTH+BLOCK_SIZE-1)/BLOCK_SIZE),BLOCK_SIZE>>>
      ( &m_blockdata->maxGridSize, static_cast<GridSizeType>(0), 1,
        &m_blockdata->values[0], static_cast<T>(0), RAJA_CUDA_REDUCE_BLOCK_LENGTH );
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceSum(const ReduceSum<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceSum<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced sum value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());

    m_blockdata->reducedValue = static_cast<T>(0);

    size_t grid_size = m_blockdata->maxGridSize;
    assert(grid_size < RAJA_CUDA_REDUCE_BLOCK_LENGTH);
    for (size_t i = 0; i < grid_size; ++i) {
      m_blockdata->reducedValue += m_blockdata->values[i];
    }
    m_reduced_val = m_init_val + m_blockdata->reducedValue;

    return m_reduced_val;
  }

  //
  // Method that returns reduced sum value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // += operator that adds value to sum in the proper device shared
  // memory block locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceSum<cuda_reduce<BLOCK_SIZE>, T> operator+=(T val) const
  {
    __shared__ T sd[BLOCK_SIZE];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    if (blockId + threadId == 0) {
      m_blockdata->maxGridSize =
          RAJA_MAX(gridDim.x * gridDim.y * gridDim.z, m_blockdata->maxGridSize);
    }

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i] = static_cast<T>(0);
      }
    }
    __syncthreads();

    sd[threadId] = val;

    T temp = 0;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] += sd[threadId + i];
      }
      __syncthreads();
    }

    if (threadId < WARP_SIZE) {
      temp = sd[threadId];
      for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        temp += shfl_xor(temp, i);
      }
    }

    // one thread adds to gmem
    if (threadId == 0) {
      m_blockdata->values[blockId] += temp;
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceSum<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;
  int m_myID;

  T m_init_val;
  T m_reduced_val;

  CudaReductionBlockType<T> *m_blockdata;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck = 
      (  (sizeof(T) <= sizeof(CudaReductionDummyDataType))
      && (sizeof(CudaReductionTallyType<T>) <= sizeof(CudaReductionDummyTallyType))
      && (sizeof(CudaReductionBlockType<T>) <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck, "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

/*!
 ******************************************************************************
 *
 * \brief  Sum reduction Atomic Non-Deterministic Variant class template
 *         for use in CUDA kernel.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceSum(T init_val)
  {
    m_is_copy = false;
    m_reduced_val = static_cast<T>(0);
    m_init_val = init_val;
    m_myID = getCudaReductionId();
    m_tallydata = static_cast<CudaReductionTallyType<T>*>(getCudaReductionTallyBlock(m_myID));

    rajaCudaMemsetType<T, T>
      <<<2, 1>>>
      ( &m_tallydata->tally, static_cast<T>(0), 1,
        &m_tallydata->initVal, init_val, 1 );
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceSum(const ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  // Destruction on host releases the global shared memory block chunk for
  //
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced sum value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaDeviceSynchronize();
    m_reduced_val = m_init_val + m_tallydata->tally;
    return m_reduced_val;
  }

  //
  // Operator that returns reduced sum value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // += operator that adds value to sum in the proper device
  // memory block locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T> operator+=(
      T val) const
  {
    __shared__ T sd[BLOCK_SIZE];

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i] = static_cast<T>(0);
      }
    }
    __syncthreads();

    sd[threadId] = val;

    T temp = 0;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] += sd[threadId + i];
      }
      __syncthreads();
    }

    if (threadId < WARP_SIZE) {
      temp = sd[threadId];
      for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        temp += shfl_xor(temp, i);
      }
    }

    // one thread adds to tally
    if (threadId == 0) {
      _atomicAdd<T>(&(m_tallydata->tally), temp);
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;

  T m_init_val;
  T m_reduced_val;

  CudaReductionTallyType<T> *m_tallydata;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck = 
      (  (sizeof(T) <= sizeof(CudaReductionDummyDataType))
      && (sizeof(CudaReductionTallyType<T>) <= sizeof(CudaReductionDummyTallyType))
      && (sizeof(CudaReductionBlockType<T>) <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck, "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

///
/// Each ReduceMinLoc or ReduceMaxLoc object uses retiredBlocks as a way
/// to complete the reduction in a single pass. Although the algorithm
/// updates retiredBlocks via an atomicAdd(int) the actual reduction values
/// do not use atomics and require a finishing stage performed
/// by the last block.
///
__device__ __managed__ GridSizeType retiredBlocks[RAJA_MAX_REDUCE_VARS];

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in a CUDA execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceMinLoc(T init_val, Index_type init_loc)
  {
    m_is_copy = false;
    m_reduced_val = init_val;
    m_reduced_idx = init_loc;
    m_myID = getCudaReductionId();
    m_blockdata = static_cast<CudaReductionLocBlockType<T>*>(getCudaReductionMemBlock(m_myID));

    CudaReductionLocType<T> tmp_init;
    tmp_init.val = init_val;
    tmp_init.idx = init_loc;

    rajaCudaMemsetType<GridSizeType, CudaReductionLocType<T>, CudaReductionLocType<T>, GridSizeType>
      <<<((1+1+RAJA_CUDA_REDUCE_BLOCK_LENGTH+1+BLOCK_SIZE-1)/BLOCK_SIZE), BLOCK_SIZE>>>
      ( &m_blockdata->maxGridSize, static_cast<GridSizeType>(0), 1,
        &m_blockdata->reducedValue, tmp_init, 1,
        &m_blockdata->values[0], tmp_init, RAJA_CUDA_REDUCE_BLOCK_LENGTH,
        &retiredBlocks[m_myID], static_cast<GridSizeType>(0), 1 );
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceMinLoc(const ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());
    size_t grid_size = m_blockdata->maxGridSize;
    assert(grid_size < RAJA_CUDA_REDUCE_BLOCK_LENGTH);
    m_reduced_val = m_blockdata->reducedValue.val;
    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // Method that returns index value corresponding to the reduced min.
  //
  // Note: accessor only executes on host.
  //
  Index_type getLoc()
  {
    cudaErrchk(cudaDeviceSynchronize());
    m_reduced_idx = m_blockdata->reducedValue.idx;
    return m_reduced_idx;
  }

  //
  // Method that updates min and index values in proper device memory block
  // locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T> minloc(
      T val,
      Index_type idx) const
  {
    __shared__ CudaReductionLocType<T> sd[BLOCK_SIZE];
    __shared__ bool lastBlock;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    if (blockId + threadId == 0) {
      m_blockdata->maxGridSize =
          RAJA_MAX(gridDim.x * gridDim.y * gridDim.z, m_blockdata->maxGridSize);
    }

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i].val = m_reduced_val;
        sd[threadId + i].idx = m_reduced_idx;
      }
    }
    __syncthreads();

    sd[threadId].val = val;
    sd[threadId].idx = idx;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + i]);
      }
      __syncthreads();
    }

    if (threadId < 16) {
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 16]);
    }
    __syncthreads();

    if (threadId < 8) {
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 8]);
    }
    __syncthreads();

    if (threadId < 4) {
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 4]);
    }
    __syncthreads();

    if (threadId < 2) {
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 2]);
    }
    __syncthreads();

    if (threadId < 1) {
      lastBlock = false;
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 1]);
      m_blockdata->values[blockId] = RAJA_MINLOC(sd[threadId], m_blockdata->values[blockId]);
      __threadfence();
      unsigned int oldBlockCount = atomicAdd((unsigned int*)&retiredBlocks[m_myID], (unsigned int)1); // use atomicInc instead
      lastBlock = (oldBlockCount == ((gridDim.x * gridDim.y * gridDim.z)- 1));
    }
    __syncthreads();

    if (lastBlock) {
      if (threadId == 0) {
        retiredBlocks[m_myID] = 0; // not necessary if using atomicInc
      }

      CudaReductionLocType<T> lmin = {m_reduced_val, m_reduced_idx};
      int blocks = gridDim.x * gridDim.y * gridDim.z;
      int threads = blockDim.x * blockDim.y * blockDim.z;
      for (int i = threadId; i < blocks; i += threads) {
        lmin = RAJA_MINLOC(lmin, m_blockdata->values[i]);
      }
      sd[threadId] = lmin;
      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + i]);
        }
        __syncthreads();
      }

      if (threadId < 16) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 16]);
      }
      __syncthreads();

      if (threadId < 8) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 8]);
      }
      __syncthreads();

      if (threadId < 4) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 4]);
      }
      __syncthreads();

      if (threadId < 2) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 2]);
      }
      __syncthreads();

      if (threadId < 1) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 1]);
        m_blockdata->reducedValue =
            RAJA_MINLOC(m_blockdata->reducedValue, sd[threadId]);
      }
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;

  T m_reduced_val;
  Index_type m_reduced_idx;

  CudaReductionLocBlockType<T> *m_blockdata;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck = 
      (  (sizeof(T) <= sizeof(CudaReductionDummyDataType))
      && (sizeof(CudaReductionLocTallyType<T>) <= sizeof(CudaReductionDummyTallyType))
      && (sizeof(CudaReductionLocBlockType<T>) <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck, "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in a CUDA execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceMaxLoc(T init_val, Index_type init_loc)
  {
    m_is_copy = false;
    m_reduced_val = init_val;
    m_reduced_idx = init_loc;
    m_myID = getCudaReductionId();
    m_blockdata = static_cast<CudaReductionLocBlockType<T>*>(getCudaReductionMemBlock(m_myID));

    CudaReductionLocType<T> tmp_init;
    tmp_init.val = init_val;
    tmp_init.idx = init_loc;

    rajaCudaMemsetType<GridSizeType, CudaReductionLocType<T>, CudaReductionLocType<T>, GridSizeType>
      <<<((1+1+RAJA_CUDA_REDUCE_BLOCK_LENGTH+1+BLOCK_SIZE-1)/BLOCK_SIZE), BLOCK_SIZE>>>
      ( &m_blockdata->maxGridSize, static_cast<GridSizeType>(0), 1,
        &m_blockdata->reducedValue, tmp_init, 1,
        &m_blockdata->values[0], tmp_init, RAJA_CUDA_REDUCE_BLOCK_LENGTH,
        &retiredBlocks[m_myID], static_cast<GridSizeType>(0), 1 );
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceMaxLoc(const ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());
    size_t grid_size = m_blockdata->maxGridSize;
    assert(grid_size < RAJA_CUDA_REDUCE_BLOCK_LENGTH);
    m_reduced_val = m_blockdata->reducedValue.val;
    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // Method that returns index value corresponding to the reduced max.
  //
  // Note: accessor only executes on host.
  //
  Index_type getLoc()
  {
    cudaErrchk(cudaDeviceSynchronize());
    m_reduced_idx = m_blockdata->reducedValue.idx;
    return m_reduced_idx;
  }

  //
  // Method that updates max and index values in proper device memory block
  // locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T> maxloc(
      T val,
      Index_type idx) const
  {
    __shared__ CudaReductionLocType<T> sd[BLOCK_SIZE];
    __shared__ bool lastBlock;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    if (blockId + threadId == 0) {
      m_blockdata->maxGridSize =
          RAJA_MAX(gridDim.x * gridDim.y * gridDim.z, m_blockdata->maxGridSize);
    }

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i].val = m_reduced_val;
        sd[threadId + i].idx = m_reduced_idx;
      }
    }
    __syncthreads();

    sd[threadId].val = val;
    sd[threadId].idx = idx;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + i]);
      }
      __syncthreads();
    }

    if (threadId < 16) {
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 16]);
    }
    __syncthreads();

    if (threadId < 8) {
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 8]);
    }
    __syncthreads();

    if (threadId < 4) {
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 4]);
    }
    __syncthreads();

    if (threadId < 2) {
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 2]);
    }
    __syncthreads();

    if (threadId < 1) {
      lastBlock = false;
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 1]);
      m_blockdata->values[blockId] = RAJA_MAXLOC(sd[threadId], m_blockdata->values[blockId]);
      __threadfence();
      unsigned int oldBlockCount = atomicAdd((unsigned int*)&retiredBlocks[m_myID], (unsigned int)1);
      lastBlock = (oldBlockCount == ((gridDim.x * gridDim.y * gridDim.z) - 1));
    }
    __syncthreads();

    if (lastBlock) {
      if (threadId == 0) {
        retiredBlocks[m_myID] = 0;
      }

      CudaReductionLocType<T> lmax = {m_reduced_val, m_reduced_idx};
      int blocks = gridDim.x * gridDim.y * gridDim.z;
      int threads = blockDim.x * blockDim.y * blockDim.z;

      for (int i = threadId; i < blocks; i += threads) {
        lmax = RAJA_MAXLOC(lmax, m_blockdata->values[i]);
      }
      sd[threadId] = lmax;
      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + i]);
        }
        __syncthreads();
      }

      if (threadId < 16) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 16]);
      }
      __syncthreads();

      if (threadId < 8) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 8]);
      }
      __syncthreads();

      if (threadId < 4) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 4]);
      }
      __syncthreads();

      if (threadId < 2) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 2]);
      }
      __syncthreads();

      if (threadId < 1) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 1]);
        m_blockdata->reducedValue =
            RAJA_MAXLOC(m_blockdata->reducedValue, sd[threadId]);
      }
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;

  T m_reduced_val;
  Index_type m_reduced_idx;

  CudaReductionLocBlockType<T> *m_blockdata;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck = 
      (  (sizeof(T) <= sizeof(CudaReductionDummyDataType))
      && (sizeof(CudaReductionLocTallyType<T>) <= sizeof(CudaReductionDummyTallyType))
      && (sizeof(CudaReductionLocBlockType<T>) <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck, "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
