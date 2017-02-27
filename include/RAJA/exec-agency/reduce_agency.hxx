/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for OpenMP
 *          OpenMP execution.
 *
 *          These methods should work on any platform that supports OpenMP.
 *
 ******************************************************************************
 */

#ifndef RAJA_reducer_agency_HXX
#define RAJA_reducer_agency_HXX

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
#include <unordered_map>

#include "RAJA/int_datatypes.hxx"

#include "RAJA/reducers.hxx"

#include "RAJA/MemUtils_CPU.hxx"

#include "RAJA/ThreadUtils_CPU.hxx"

#include "agency/agency.hpp"

#include <iostream>

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMin<agency_reduce, T>
{
public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMin(T init_val)
  {
    m_is_copy = false;

    m_reduced_val = init_val;

    m_myID = getCPUReductionId();

    m_blockdata = getCPUReductionMemBlock(m_myID);

    //Creating a mapping of thread ids to integer numbers
    threadMap = std::make_shared<std::unordered_map<std::thread::id, int>>();

    //Create mutex
    mtx = std::make_shared<std::mutex>();

    m_init_val = init_val;
  }

  //
  // Copy ctor.
  //
  ReduceMin(const ReduceMin<agency_reduce, T>& other)
  {
    //add the ID to the thread map if it is not already added
    
    *this = other;
    m_is_copy = true;
    
    mtx->lock();
    auto search =  threadMap->find(std::this_thread::get_id());

    if (search == threadMap->end()){
      (*threadMap)[std::this_thread::get_id()] = (threadMap)->size()-1;
      //Initilize the value for this thread
      m_blockdata[ (*threadMap)[std::this_thread::get_id()] * s_block_offset] = m_init_val;
    } 
    mtx->unlock();
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMin<agency_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    for (int i = 0; i < threadMap->size(); ++i) {
      m_reduced_val = RAJA_MIN(m_reduced_val,
                               static_cast<T>(m_blockdata[i * s_block_offset]));
    }

    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  T get() { return operator T(); }

  //
  // Method that updates min value for current thread.
  //
  ReduceMin<agency_reduce, T> min(T val) const
  {
    mtx->lock();
    int tid = (*threadMap)[std::this_thread::get_id()];
    mtx->unlock();

    int idx = tid * s_block_offset;
    m_blockdata[idx] = RAJA_MIN(static_cast<T>(m_blockdata[idx]), val);

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMin<agency_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);

  bool m_is_copy;
  int m_myID;

  std::shared_ptr<std::mutex> mtx; 

  std::shared_ptr<std::unordered_map<std::thread::id, int> > threadMap;

  T m_reduced_val;

  T m_init_val;

  CPUReductionBlockDataType* m_blockdata;
};

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMinLoc<agency_reduce, T>
{
public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMinLoc(T init_val, Index_type init_loc)
  {
    m_is_copy = false;

    m_reduced_val = init_val;

    m_myID = getCPUReductionId();

    m_blockdata = getCPUReductionMemBlock(m_myID);
    m_idxdata = getCPUReductionLocBlock(m_myID);

    m_init_val = init_val;
    m_init_loc = init_loc;


    //Creating a mapping of thread ids to integer numbers
    threadMap = std::make_shared<std::unordered_map<std::thread::id, int>>();
    mtx = std::make_shared<std::mutex>();

  }

  //
  // Copy ctor.
  //
  ReduceMinLoc(const ReduceMinLoc<agency_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;

    //add the ID to the thread map if it is not already added
    mtx->lock();
    auto search =  threadMap->find(std::this_thread::get_id());
    if (search == (threadMap)->end()){
      (*threadMap)[std::this_thread::get_id()] = (threadMap)->size()-1;
      m_blockdata[(*threadMap)[std::this_thread::get_id()] * s_block_offset] = m_init_val;
      m_idxdata[(*threadMap)[std::this_thread::get_id()] * s_idx_offset] = m_init_loc;

    } 
    mtx->unlock();

  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMinLoc<agency_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    if (threadMap->size() == 0){
      return m_init_val;
    }

    for (int i = 0; i < threadMap->size(); ++i) {
      if (static_cast<T>(m_blockdata[i * s_block_offset]) <= m_reduced_val) {
        m_reduced_val = m_blockdata[i * s_block_offset];
        m_reduced_idx = m_idxdata[i * s_idx_offset];
      }
    }

    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced min value.
  //
  Index_type getLoc()
  {
    if (threadMap->size() == 0){
      return m_init_loc;
    }

    for (int i = 0; i < threadMap->size(); ++i) {
      if (static_cast<T>(m_blockdata[i * s_block_offset]) <= m_reduced_val) {
        m_reduced_val = m_blockdata[i * s_block_offset];
        m_reduced_idx = m_idxdata[i * s_idx_offset];
      }
    }

    return m_reduced_idx;
  }

  //
  // Method that updates min and index values for current thread.
  //
  ReduceMinLoc<agency_reduce, T> minloc(T val, Index_type idx) const
  {
    mtx->lock();
    int tid = (*threadMap)[std::this_thread::get_id()];
    mtx->unlock();

    if (val <= static_cast<T>(m_blockdata[tid * s_block_offset])) {
      m_blockdata[tid * s_block_offset] = val;
      m_idxdata[tid * s_idx_offset] = idx;
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMinLoc<agency_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);
  static const int s_idx_offset = COHERENCE_BLOCK_SIZE / sizeof(Index_type);

  bool m_is_copy;
  int m_myID;

  T m_init_val;
  Index_type m_init_loc;

  std::shared_ptr<std::mutex> mtx; 

  std::shared_ptr<std::unordered_map<std::thread::id, int> > threadMap;

  T m_reduced_val;
  Index_type m_reduced_idx;

  CPUReductionBlockDataType* m_blockdata;
  Index_type* m_idxdata;
};

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMax<agency_reduce, T>
{
public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMax(T init_val)
  {
    m_is_copy = false;

    m_reduced_val = init_val;

    m_myID = getCPUReductionId();

    m_blockdata = getCPUReductionMemBlock(m_myID);

    m_init_val = init_val;

    //Creating a mapping of thread ids to integer numbers
    threadMap = std::make_shared<std::unordered_map<std::thread::id, int>>();

    //create mutex
    mtx = std::make_shared<std::mutex>();


  }

  //
  // Copy ctor.
  //
  ReduceMax(const ReduceMax<agency_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;

    //add the ID to the thread map if it is not already added
    mtx->lock();
    auto search =  threadMap->find(std::this_thread::get_id());
    if (search == threadMap->end()){
      (*threadMap)[std::this_thread::get_id()] = (threadMap)->size() -1;
      m_blockdata[(*threadMap)[std::this_thread::get_id()] * s_block_offset] = m_init_val;

    } 
    mtx->unlock();

  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMax<agency_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    for (int i = 0; i < threadMap->size(); ++i) {
      m_reduced_val = RAJA_MAX(m_reduced_val,
                               static_cast<T>(m_blockdata[i * s_block_offset]));
    }

    return m_reduced_val;
  }

  //
  // Method that returns reduced max value.
  //
  T get() { return operator T(); }

  //
  // Method that updates max value for current thread.
  //
  ReduceMax<agency_reduce, T> max(T val) const
  {
    mtx->lock();
    int tid = (*threadMap)[std::this_thread::get_id()];
    mtx->unlock();

    int idx = tid * s_block_offset;
    m_blockdata[idx] = RAJA_MAX(static_cast<T>(m_blockdata[idx]), val);

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMax<agency_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);

  bool m_is_copy;
  int m_myID;

  T m_init_val;

  std::shared_ptr<std::mutex> mtx; 

  std::shared_ptr<std::unordered_map<std::thread::id, int> > threadMap;

  T m_reduced_val;

  CPUReductionBlockDataType* m_blockdata;
};

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMaxLoc<agency_reduce, T>
{
public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMaxLoc(T init_val, Index_type init_loc)
  {
    m_is_copy = false;

    m_reduced_val = init_val;

    m_myID = getCPUReductionId();

    m_blockdata = getCPUReductionMemBlock(m_myID);
    m_idxdata = getCPUReductionLocBlock(m_myID);

    m_init_val = init_val;
    m_init_loc = init_loc;

    //Creating a mapping of thread ids to integer numbers
    threadMap = std::make_shared<std::unordered_map<std::thread::id, int>>();

    //create mutex
    mtx = std::make_shared<std::mutex>();

    //add this thread to the map
    mtx->lock();
    (*threadMap)[std::this_thread::get_id()] = 0;
    mtx->unlock();

  }

  //
  // Copy ctor.
  //
  ReduceMaxLoc(const ReduceMaxLoc<agency_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;

    //add the ID to the thread map if it is not already added
    mtx->lock();
    auto search =  threadMap->find(std::this_thread::get_id());
    if (search == (threadMap)->end()){
      (*threadMap)[std::this_thread::get_id()] = (threadMap)->size()-1;
      m_blockdata[(*threadMap)[std::this_thread::get_id()] * s_block_offset] = m_init_val;
      m_idxdata[(*threadMap)[std::this_thread::get_id()] * s_idx_offset] = m_init_loc;

    } 
    mtx->unlock();

  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMaxLoc<agency_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    if (threadMap->size()==0){
      return m_init_val;
    }
    for (int i = 0; i < threadMap->size(); ++i) {
      if (static_cast<T>(m_blockdata[i * s_block_offset]) >= m_reduced_val) {
        m_reduced_val = m_blockdata[i * s_block_offset];
        m_reduced_idx = m_idxdata[i * s_idx_offset];
      }
    }

    return m_reduced_val;
  }

  //
  // Method that returns reduced max value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced max value.
  //
  Index_type getLoc()
  {
    if (threadMap->size()==0){
      return m_init_loc;
    }

    for (int i = 0; i < threadMap->size(); ++i) {
      if (static_cast<T>(m_blockdata[i * s_block_offset]) >= m_reduced_val) {
        m_reduced_val = m_blockdata[i * s_block_offset];
        m_reduced_idx = m_idxdata[i * s_idx_offset];
      }
    }

    return m_reduced_idx;
  }

  //
  // Method that updates max and index values for current thread.
  //
  ReduceMaxLoc<agency_reduce, T> maxloc(T val, Index_type idx) const
  {
    mtx->lock();
    int tid = (*threadMap)[std::this_thread::get_id()];
    mtx->unlock();

    if (val >= static_cast<T>(m_blockdata[tid * s_block_offset])) {
      m_blockdata[tid * s_block_offset] = val;
      m_idxdata[tid * s_idx_offset] = idx;
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMaxLoc<agency_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);
  static const int s_idx_offset = COHERENCE_BLOCK_SIZE / sizeof(Index_type);

  bool m_is_copy;
  int m_myID;

  T m_init_val;
  Index_type m_init_loc;

  std::shared_ptr<std::mutex> mtx; 

  std::shared_ptr<std::unordered_map<std::thread::id, int> > threadMap;

  T m_reduced_val;
  Index_type m_reduced_idx;

  CPUReductionBlockDataType* m_blockdata;
  Index_type* m_idxdata;
};

/*!
 ******************************************************************************
 *
 * \brief  Sum reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceSum<agency_reduce, T>
{
public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceSum(T init_val)
  {
    m_is_copy = false;

    m_init_val = init_val;
    m_reduced_val = static_cast<T>(0);

    m_myID = getCPUReductionId();

    m_blockdata = getCPUReductionMemBlock(m_myID);


    //Creating a mapping of thread ids to integer numbers
    threadMap = std::make_shared<std::unordered_map<std::thread::id, int>>();

    //create mutex
    mtx = std::make_shared<std::mutex>();
    m_blockdata[0] = m_init_val;



  }

  //
  // Copy ctor.
  //
  ReduceSum(const ReduceSum<omp_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;

    //add the ID to the thread map if it is not already added
    mtx->lock();
    auto search =  threadMap->find(std::this_thread::get_id());
    if (search == (threadMap)->end()){
      (*threadMap)[std::this_thread::get_id()] = (threadMap)->size()-1;
      m_blockdata[(*threadMap)[std::this_thread::get_id()]* s_block_offset] = m_init_val;
    } 
    mtx->unlock();

  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceSum<agency_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced sum value.
  //
  operator T()
  {
    T tmp_reduced_val = static_cast<T>(0);
    for (int i = 0; i < threadMap->size(); ++i) {
      tmp_reduced_val += static_cast<T>(m_blockdata[i * s_block_offset]);
    }
    m_reduced_val = m_init_val + tmp_reduced_val;

    return m_reduced_val;
  }

  //
  // Method that returns sum value.
  //
  T get() { return operator T(); }

  //
  // += operator that adds value to sum for current thread.
  //
  ReduceSum<agency_reduce, T> operator+=(T val) const
  {

    mtx->lock();
    if (threadMap->find(std::this_thread::get_id()) == (threadMap)->end()){
      (*threadMap)[std::this_thread::get_id()] = (threadMap)->size()-1;
      m_blockdata[(*threadMap)[std::this_thread::get_id()]* s_block_offset] = m_init_val;
    }
    int tid = (*threadMap)[std::this_thread::get_id()];
   
   mtx->unlock(); 

    m_blockdata[tid * s_block_offset] += val;
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceSum<agency_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);

  bool m_is_copy;
  int m_myID;

  std::shared_ptr<std::mutex> mtx; 

  std::shared_ptr<std::unordered_map<std::thread::id, int> > threadMap;

  T m_init_val;
  T m_reduced_val;

  CPUReductionBlockDataType* m_blockdata;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for agency include guard
#endif  // closing endif for header file include guard
