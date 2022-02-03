
#ifndef NARF_TRAITS_H
#define NARF_TRAITS_H


#include <boost/histogram.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace narf {

template <typename T>
struct acc_traits {
  static constexpr bool is_weighted_sum = false;
  static constexpr bool is_tensor = false;
//   using value_type = T;
};

template <typename T>
struct acc_traits<boost::histogram::accumulators::weighted_sum<T>> {
  static constexpr bool is_weighted_sum = true;
  static constexpr bool is_tensor = false;
//   using value_type = T;
};

template <typename T>
struct storage_traits {
  static constexpr bool is_adopted = false;
};

template <typename T, typename Enable = void>
struct tensor_traits {
  static constexpr bool is_container = false;
  static constexpr bool is_tensor = false;
};

template <typename T, int Options_, typename IndexType, std::ptrdiff_t... Indices>
struct tensor_traits<Eigen::TensorFixedSize<T, Eigen::Sizes<Indices...>, Options_, IndexType>> {
  static constexpr bool is_container = false;
  static constexpr bool is_tensor = true;
  static constexpr std::size_t rank = sizeof...(Indices);
  static constexpr ptrdiff_t size = (Indices*...*static_cast<ptrdiff_t>(1));
  static constexpr std::array<std::ptrdiff_t, sizeof...(Indices)> sizes = { Indices... };
  using value_type = T;

  // needed for PyROOT/cppyy since it can't currently handle the static constexpr member directly
  static constexpr std::array<std::ptrdiff_t, sizeof...(Indices)> get_sizes() { return sizes; }
};

template <typename T>
struct tensor_traits<T, std::enable_if_t<ROOT::Internal::RDF::IsDataContainer<T>::value>> : public tensor_traits<typename T::value_type> {
  static constexpr bool is_container = true;
};

template <typename T>
struct is_array_interface_view : public std::false_type{
};

#if __cplusplus < 202002L
template< class T >
struct type_identity {
    using type = T;
};
#else
template< class T >
using type_identity = std::type_identity<T>;
#endif

template< class T >
using type_identity_t = typename type_identity<T>::type;

}

#endif
