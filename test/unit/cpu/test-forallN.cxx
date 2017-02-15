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

#include <time.h>
#include <cmath>
#include <cmath>
#include <cstdlib>

#include <iostream>
#include <string>
#include <vector>
#include <random>

#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"
#include "RAJA/internal/defines.hxx"

#include "Compare.hxx"

template <typename T>
class ForallNCorrectnessTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(ForallNCorrectnessTest);

TYPED_TEST_P(ForallNCorrectnessTest, TwoDimensionTest)
{
  using View = typename std::tuple_element<0, TypeParam>::type;
  using ExecutionPolicy = typename std::tuple_element<1, TypeParam>::type;

  int sizes[2][2] = { { 128, 1024 }, { 37, 1 } };

  for (int size_index = 0; size_index < 2; ++size_index) {
      auto size = sizes[size_index];

      int size_i = size[0];
      int size_j = size[1];
      std::vector<int> values(size_i * size_j, 1);

      View val_view(&values[0], size_i, size_j);

      RAJA::forallN<ExecutionPolicy>(
          RAJA::RangeSegment(1, size_i),
          RAJA::RangeSegment(0, size_j),
          [=](RAJA::Index_type i, RAJA::Index_type j) {
              val_view(0, j) += val_view(i, j);
          });

      RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>>(
          RAJA::RangeSegment(0, size_i),
          RAJA::RangeSegment(0, size_j),
          [&](RAJA::Index_type i, RAJA::Index_type j) {
            if (i == 0) {
              EXPECT_EQ(val_view(i, j), size_i);
            } else {
              EXPECT_EQ(val_view(i, j), 1);
            }
          });
  }
}

REGISTER_TYPED_TEST_CASE_P(ForallNCorrectnessTest, TwoDimensionTest);

using IJ_VIEW = RAJA::View<int, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
using JI_VIEW = RAJA::View<int, RAJA::Layout<int, RAJA::PERM_JI, int, int>>;

using BasicSequentialTypes = ::testing::Types<
  std::tuple<
      IJ_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>,
          RAJA::Permute<RAJA::PERM_IJ>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::seq_exec>, 
          RAJA::Permute<RAJA::PERM_JI>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::seq_exec>,
          RAJA::Tile<
              RAJA::TileList<RAJA::tile_fixed<8>, RAJA::tile_fixed<16>>,
              RAJA::Permute<RAJA::PERM_JI>>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::seq_exec>,
          RAJA::Tile<
              RAJA::TileList<RAJA::tile_fixed<32>, RAJA::tile_fixed<32>>,
              RAJA::Tile<
                  RAJA::TileList<RAJA::tile_fixed<8>, RAJA::tile_fixed<16>>,
                  RAJA::Permute<RAJA::PERM_JI>>>>>>;
                  
INSTANTIATE_TYPED_TEST_CASE_P(Sequential, ForallNCorrectnessTest, BasicSequentialTypes);

#ifdef RAJA_ENABLE_OPENMP
using BasicOpenMPTypes = ::testing::Types<
  std::tuple<
      IJ_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::seq_exec, RAJA::omp_parallel_for_exec>,
          RAJA::Permute<RAJA::PERM_IJ>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::omp_for_nowait_exec>,
          RAJA::OMP_Parallel<
              RAJA::Permute<RAJA::PERM_JI>>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::omp_for_nowait_exec>,
          RAJA::OMP_Parallel<
              RAJA::Tile<
                  RAJA::TileList<RAJA::tile_fixed<8>, RAJA::tile_fixed<16>>,
                  RAJA::Permute<RAJA::PERM_JI>>>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::omp_for_nowait_exec>,
          RAJA::OMP_Parallel<
              RAJA::Tile<
                  RAJA::TileList<RAJA::tile_fixed<32>, RAJA::tile_fixed<32>>,
                  RAJA::Tile<
                      RAJA::TileList<RAJA::tile_fixed<8>, RAJA::tile_fixed<16>>,
                      RAJA::Permute<RAJA::PERM_JI>>>>>>>;

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, ForallNCorrectnessTest, BasicOpenMPTypes);
#endif // defined(RAJA_ENABLE_OPENMP)

#if defined(RAJA_ENABLE_AGENCY) 
using BasicAgencyTypes = ::testing::Types<
  std::tuple<
      IJ_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::seq_exec, RAJA::agency_parallel_exec>,
          RAJA::Permute<RAJA::PERM_IJ>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::agency_parallel_exec>, 
          RAJA::Agency_Parallel<
              RAJA::agency_parallel_exec::Agent_t,
              RAJA::agency_parallel_exec::Worker_t,
              RAJA::Permute<RAJA::PERM_JI>>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::agency_parallel_exec>,
          RAJA::Agency_Parallel<
              RAJA::agency_parallel_exec::Agent_t,
              RAJA::agency_parallel_exec::Worker_t,
              RAJA::Tile<
                  RAJA::TileList<RAJA::tile_fixed<8>, RAJA::tile_fixed<16>>,
                  RAJA::Permute<RAJA::PERM_JI>>>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::agency_parallel_exec>,
          RAJA::Agency_Parallel<
              RAJA::agency_parallel_exec::Agent_t,
              RAJA::agency_parallel_exec::Worker_t,
              RAJA::Tile<
                  RAJA::TileList<RAJA::tile_fixed<32>, RAJA::tile_fixed<32>>,
                  RAJA::Tile<
                      RAJA::TileList<RAJA::tile_fixed<8>, RAJA::tile_fixed<16>>,
                      RAJA::Permute<RAJA::PERM_JI>>>>>>>;

INSTANTIATE_TYPED_TEST_CASE_P(Agency, ForallNCorrectnessTest, BasicAgencyTypes);

#if defined(RAJA_ENABLE_OPENMP)
using BasicAgencyOpenMPTypes = ::testing::Types<
  std::tuple<
      IJ_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::seq_exec, RAJA::agency_omp_parallel_exec>,
          RAJA::Permute<RAJA::PERM_IJ>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::agency_omp_parallel_exec>, 
          RAJA::Agency_Parallel<
              RAJA::agency_omp_parallel_exec::Agent_t,
              RAJA::agency_omp_parallel_exec::Worker_t,
              RAJA::Permute<RAJA::PERM_JI>>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::agency_omp_parallel_exec>,
          RAJA::Agency_Parallel<
              RAJA::agency_omp_parallel_exec::Agent_t,
              RAJA::agency_omp_parallel_exec::Worker_t,
              RAJA::Tile<
                  RAJA::TileList<RAJA::tile_fixed<8>, RAJA::tile_fixed<16>>,
                  RAJA::Permute<RAJA::PERM_JI>>>>>,
  std::tuple<
      JI_VIEW,
      RAJA::NestedPolicy<
          RAJA::ExecList<RAJA::simd_exec, RAJA::agency_omp_parallel_exec>,
          RAJA::Agency_Parallel<
              RAJA::agency_omp_parallel_exec::Agent_t,
              RAJA::agency_omp_parallel_exec::Worker_t,
              RAJA::Tile<
                  RAJA::TileList<RAJA::tile_fixed<32>, RAJA::tile_fixed<32>>,
                  RAJA::Tile<
                      RAJA::TileList<RAJA::tile_fixed<8>, RAJA::tile_fixed<16>>,
                      RAJA::Permute<RAJA::PERM_JI>>>>>>>;

INSTANTIATE_TYPED_TEST_CASE_P(AgencyOpenMP, ForallNCorrectnessTest, BasicAgencyOpenMPTypes);
#endif // defined(RAJA_ENABLE_OPENMP)
#endif // defined(RAJA_ENABLE_AGENCY)

///////////////////////////////////////////////////////////////////////////
//
// Example LTimes kernel test routines
//
// Demonstrates a 4-nested loop, the use of complex nested policies and
// the use of strongly-typed indices
//
// This routine computes phi(m, g, z) = SUM_d {  ell(m, d)*psi(d,g,z)  }
//
///////////////////////////////////////////////////////////////////////////
//
RAJA_INDEX_VALUE(IMoment, "IMoment");
RAJA_INDEX_VALUE(IDirection, "IDirection");
RAJA_INDEX_VALUE(IGroup, "IGroup");
RAJA_INDEX_VALUE(IZone, "IZone");

template <typename T>
class ForallNKernelTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(ForallNKernelTest);

TYPED_TEST_P(ForallNKernelTest, LTimesKernel)
{
  int sizes[2][4] = { { 25, 96, 8, 32 }, { 100, 15, 7, 13 } };

  using EllView = typename std::tuple_element<0, TypeParam>::type;
  using PsiView = typename std::tuple_element<1, TypeParam>::type;
  using PhiView = typename std::tuple_element<2, TypeParam>::type;
  using Exec = typename std::tuple_element<3, TypeParam>::type;

  for (int size_index = 0; size_index < 2; ++size_index) {
      auto size = sizes[size_index];
      RAJA::Index_type num_moments = size[0];
      RAJA::Index_type num_directions = size[1];
      RAJA::Index_type num_groups = size[2];
      RAJA::Index_type num_zones = size[3];

      std::vector<double> ell_data(num_moments * num_directions);
      std::vector<double> psi_data(num_directions * num_groups * num_zones);
      std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);

      std::random_device rand;
      std::mt19937 gen(rand());
      std::uniform_real_distribution<double> rand_gen(0.0,1.0);

      for (size_t i = 0; i < ell_data.size(); ++i) {
        ell_data[i] = rand_gen(gen);
      }
      for (size_t i = 0; i < psi_data.size(); ++i) {
        psi_data[i] = rand_gen(gen);
      }

      EllView ell(&ell_data[0], num_moments, num_directions);
      PsiView psi(&psi_data[0], num_directions, num_groups, num_zones);
      PhiView phi(&phi_data[0], num_moments, num_groups, num_zones);

      RAJA::forallN<Exec, IMoment, IDirection, IGroup, IZone>(
          RAJA::RangeSegment(0, num_moments),
          RAJA::RangeSegment(0, num_directions),
          RAJA::RangeSegment(0, num_groups),
          RAJA::RangeSegment(0, num_zones),
          [=](IMoment m, IDirection d, IGroup g, IZone z) {
            phi(m, g, z) += ell(m, d) * psi(d, g, z);
          });

      for (IZone z(0); z < num_zones; ++z) {
        for (IGroup g(0); g < num_groups; ++g) {
          for (IMoment m(0); m < num_moments; ++m) {
            double total = 0.0;
            for (IDirection d(0); d < num_directions; ++d) {
              total += ell(m, d) * psi(d, g, z);
            }

            EXPECT_NEAR(total, phi(m, g, z), 1e-12);
          }
        }
      }
  }
}

REGISTER_TYPED_TEST_CASE_P(ForallNKernelTest, LTimesKernel);

// Order: Ell, Psi, Phi, Exec
using LTimesSequentialTypes = ::testing::Types<
    std::tuple<
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, IMoment, IDirection>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJK, IDirection, IGroup, IZone>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJK, IMoment, IGroup, IZone>>,
        RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>>>,
    std::tuple<
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_JI, IMoment, IDirection>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_KJI, IDirection, IGroup, IZone>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJK, IMoment, IGroup, IZone>>,
        RAJA::NestedPolicy<
            RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>,
            RAJA::Permute<RAJA::PERM_LKJI>>>,
    std::tuple<
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, IMoment, IDirection>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJK, IDirection, IGroup, IZone>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_KJI, IMoment, IGroup, IZone>>,
        RAJA::NestedPolicy<
            RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>,
            RAJA::Tile<
                RAJA::TileList<RAJA::tile_none, RAJA::tile_none,
                               RAJA::tile_fixed<64>, RAJA::tile_fixed<64>>,
                RAJA::Permute<RAJA::PERM_JKIL>>>>>;
                
INSTANTIATE_TYPED_TEST_CASE_P(Sequential, ForallNKernelTest, LTimesSequentialTypes); 

#ifdef RAJA_ENABLE_OPENMP
using LTimesOpenMPTypes = ::testing::Types<
    std::tuple<
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, IMoment, IDirection>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_KJI, IDirection, IGroup, IZone>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_KJI, IMoment, IGroup, IZone>>,
        RAJA::NestedPolicy<
            RAJA::ExecList<
                RAJA::seq_exec, RAJA::seq_exec, 
                RAJA::seq_exec, RAJA::omp_for_nowait_exec>,
            RAJA::OMP_Parallel<RAJA::Permute<RAJA::PERM_LKIJ>>>>,
    std::tuple<
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, IMoment, IDirection>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_KJI, IDirection, IGroup, IZone>>,
        RAJA::View<double, RAJA::Layout<int, RAJA::PERM_KJI, IMoment, IGroup, IZone>>,
        RAJA::NestedPolicy<
            RAJA::ExecList<
                RAJA::seq_exec, RAJA::seq_exec, 
                RAJA::omp_collapse_nowait_exec, RAJA::omp_collapse_nowait_exec>,
            RAJA::OMP_Parallel<
                RAJA::Tile<
                    RAJA::TileList<RAJA::tile_none, RAJA::tile_none,
                                   RAJA::tile_none, RAJA::tile_fixed<16>>,
                    RAJA::Permute<
                        RAJA::PERM_LKIJ, 
                        RAJA::Execute // implict
                        >>>>>>;

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, ForallNKernelTest, LTimesOpenMPTypes); 
#endif

#if defined(RAJA_ENABLE_AGENCY)
// TODO: Write code here, depends on permute and tiling impl
#if defined(RAJA_ENABLE_OPENMP)

#endif
#endif
