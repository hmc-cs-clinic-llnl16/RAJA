#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>

#include <cstdlib>

#include <gtest/gtest.h>

#include "RAJA/RAJA.hxx"
#include "RAJA/internal/defines.hxx"

#include "data_storage.hxx"
#include "type_helper.hxx"

const int N = 1024;

// Unit Test Space Exploration

// We have to do all of them separately like this so the
// Cross works; otherwise we end up with too many template arguments
// and it kills the compiler
//
// Doing all of these _drastically_ increases compilation time.
using SequentialExecTypes = std::tuple<RAJA::seq_exec>;
#if defined(RAJA_ENABLE_OPENMP)
using OpenMPExecTypes = std::tuple<RAJA::omp_parallel_for_exec>;
#endif
#if defined(RAJA_ENABLE_AGENCY)
using AgencyExecTypes = std::tuple<RAJA::agency_parallel_exec>;
#if defined(RAJA_ENABLE_OPENMP)
using AgencyOpenMPExecTypes = std::tuple<RAJA::agency_omp_parallel_exec>;
#endif
#endif

using ReduceTypes = std::tuple<RAJA::operators::safe_plus<int>,
                               RAJA::operators::safe_plus<float>,
                               RAJA::operators::safe_plus<double>,
                               RAJA::operators::minimum<int>,
                               RAJA::operators::minimum<float>,
                               RAJA::operators::minimum<double>,
                               RAJA::operators::maximum<int>,
                               RAJA::operators::maximum<float>,
                               RAJA::operators::maximum<double>>;

using InPlaceTypes = std::tuple<std::false_type, std::true_type>;

template <typename T1, typename T2>
using Cross = typename types::product<T1, T2>;

using SequentialTypes = Cross<Cross<SequentialExecTypes, ReduceTypes>::type, InPlaceTypes>::type;
#if defined(RAJA_ENABLE_OPENMP)
using OpenMPTypes = Cross<Cross<OpenMPExecTypes, ReduceTypes>::type, InPlaceTypes>::type;
#endif
#if defined(RAJA_ENABLE_AGENCY)
using AgencyTypes = Cross<Cross<AgencyExecTypes, ReduceTypes>::type, InPlaceTypes>::type;
#if defined(RAJA_ENABLE_OPENMP)
using AgencyOpenMPTypes = Cross<Cross<AgencyOpenMPExecTypes, ReduceTypes>::type, InPlaceTypes>::type;
#endif
#endif

template <typename T>
struct ForTesting {
};

template <typename... Ts>
struct ForTesting<std::tuple<Ts...>> {
  using type = ::testing::Types<Ts...>;
};

using SequentialCrossTypes = ForTesting<SequentialTypes>::type;
#if defined(RAJA_ENABLE_OPENMP)
using OpenMPCrossTypes = ForTesting<OpenMPTypes>::type;
#endif
#if defined(RAJA_ENABLE_AGENCY)
using AgencyCrossTypes = ForTesting<AgencyTypes>::type;
#if defined(RAJA_ENABLE_OPENMP)
using AgencyOpenMPCrossTypes = ForTesting<AgencyOpenMPTypes>::type;
#endif
#endif


// dispatchers

template <typename Exec, typename Fn, typename Storage>
void inclusive(Storage* data, bool inPlace = false)
{
  if (inPlace)
    RAJA::inclusive_scan_inplace<Exec>(data->ibegin(), data->iend(), Fn{});
  else
    RAJA::inclusive_scan<Exec>(data->ibegin(),
                               data->iend(),
                               data->obegin(),
                               Fn{});
}

template <typename Exec, typename Fn, typename Storage>
void exclusive(Storage* data, bool inPlace = false)
{
  if (inPlace)
    RAJA::exclusive_scan_inplace<Exec>(data->ibegin(), data->iend(), Fn{});
  else
    RAJA::exclusive_scan<Exec>(data->ibegin(),
                               data->iend(),
                               data->obegin(),
                               Fn{});
}

// comparators

template <typename Data,
          typename Storage,
          typename Fn,
          typename T = typename std::remove_pointer<Data>::type::type>
void compareInclusive(Data original, Storage data, Fn function, T RAJA_UNUSED_ARG(init))
{
  auto in = original->ibegin();
  auto out = data->obegin();
  T sum = *in;
  int index = 0;
  while ((out + index) != data->oend()) {
    ASSERT_EQ(sum, *(out + index)) << "Expected value differs at index "
                                   << index;
    ++index;
    sum = function(sum, *(in + index));
  }
}

template <typename Data,
          typename Storage,
          typename Fn,
          typename T = typename std::remove_pointer<Data>::type::type>
void compareExclusive(Data original, Storage data, Fn function, T init)
{
  auto in = original->ibegin();
  auto out = data->obegin();
  T sum = init;
  int index = 0;
  while ((out + index) != data->oend()) {
    ASSERT_EQ(sum, *(out + index)) << "Expected value differs at index "
                                   << index;
    sum = function(sum, *(in + index));
    ++index;
  }
}

// test implementations

template <typename Tuple>
class ScanTest : public testing::Test
{
public:
  using Exec = typename std::tuple_element<0, Tuple>::type;
  using Fn = typename std::tuple_element<1, Tuple>::type;
  using Bool = typename std::tuple_element<2, Tuple>::type;
  constexpr const static bool InPlace = Bool::value;
  using T = typename Fn::result_type;
  using Storage = storage<Exec, T, InPlace>;

protected:
  virtual void SetUp()
  {
    std::iota(data->ibegin(), data->iend(), 1);
    std::shuffle(data->ibegin(),
                 data->iend(),
                 std::mt19937{std::random_device{}()});
    std::copy(data->ibegin(), data->iend(), original->ibegin());
  }

  std::unique_ptr<Storage> data = std::unique_ptr<Storage>(new Storage{N});
  std::unique_ptr<Storage> original = std::unique_ptr<Storage>(new Storage{N});
  Fn function = Fn{};
};

template <typename Tuple>
class InclusiveScanTest : public ScanTest<Tuple>
{
protected:
  virtual void SetUp()
  {
    ScanTest<Tuple>::SetUp();
    inclusive<typename ScanTest<Tuple>::Exec, typename ScanTest<Tuple>::Fn>(
        this->data.get(), ScanTest<Tuple>::InPlace);
  }
};

template <typename Tuple>
class ExclusiveScanTest : public ScanTest<Tuple>
{
protected:
  virtual void SetUp()
  {
    ScanTest<Tuple>::SetUp();
    exclusive<typename ScanTest<Tuple>::Exec, typename ScanTest<Tuple>::Fn>(
        this->data.get(), ScanTest<Tuple>::InPlace);
  }
};

TYPED_TEST_CASE_P(InclusiveScanTest);
TYPED_TEST_CASE_P(ExclusiveScanTest);

TYPED_TEST_P(InclusiveScanTest, InclusiveCorrectness)
{
  auto init = decltype(this->function)::identity;
  compareInclusive(this->original.get(),
                   this->data.get(),
                   this->function,
                   init);
}
TYPED_TEST_P(ExclusiveScanTest, ExclusiveCorrectness)
{
  auto init = decltype(this->function)::identity;
  compareExclusive(this->original.get(),
                   this->data.get(),
                   this->function,
                   init);
}

REGISTER_TYPED_TEST_CASE_P(InclusiveScanTest, InclusiveCorrectness);
REGISTER_TYPED_TEST_CASE_P(ExclusiveScanTest, ExclusiveCorrectness);

INSTANTIATE_TYPED_TEST_CASE_P(SequentialScan, InclusiveScanTest, SequentialCrossTypes);
INSTANTIATE_TYPED_TEST_CASE_P(SequentialScan, ExclusiveScanTest, SequentialCrossTypes);
#if defined(RAJA_ENABLE_OPENMP)
INSTANTIATE_TYPED_TEST_CASE_P(OpenMPScan, InclusiveScanTest, OpenMPCrossTypes);
INSTANTIATE_TYPED_TEST_CASE_P(OpenMPScan, ExclusiveScanTest, OpenMPCrossTypes);
#endif
#if defined(RAJA_ENABLE_AGENCY)
INSTANTIATE_TYPED_TEST_CASE_P(AgencyScan, InclusiveScanTest, AgencyCrossTypes);
INSTANTIATE_TYPED_TEST_CASE_P(AgencyScan, ExclusiveScanTest, AgencyCrossTypes);
#if defined(RAJA_ENABLE_OPENMP)
INSTANTIATE_TYPED_TEST_CASE_P(AgencyOpenMPScan, InclusiveScanTest, AgencyOpenMPCrossTypes);
INSTANTIATE_TYPED_TEST_CASE_P(AgencyOpenMPScan, ExclusiveScanTest, AgencyOpenMPCrossTypes);
#endif
#endif
