
#ifndef NARF_TENSORUTILS_H
#define NARF_TENSORUTILS_H

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace narf {

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
class TensorAccumulator {
public:

  using tensor_t = T;

  static constexpr std::ptrdiff_t size() { return tensor_traits<T>::size; }

  // prefix increment
  TensorAccumulator &operator++()
  {
    for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it) {
      ++(*it);
    }
    return *this;
  }

  TensorAccumulator &operator+=(const boost::histogram::weight_type<const T&> &rhs)
  {
    const typename T::Scalar *itr = rhs.value.data();
    for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it, ++itr) {
      (*it) += *itr;
    }
    return *this;
  }

  // don't use T::operator+= because this is not safe for use with atomics, being implemented as
  // lhs = lhs + rhs
  template <typename U>
  TensorAccumulator &operator+=(const U &rhs) {
    const typename U::Scalar *itr = rhs.data();
    for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it, ++itr) {
      (*it) += *itr;
    }
    return *this;
  }

  template <typename... Us>
  typename T::Scalar &operator() (Us&&... us) { return tensor_(std::forward<Us>(us)...); }

  template <typename... Us>
  const typename T::Scalar &operator() (Us&&... us) const { return tensor_(std::forward<Us>(us)...); }

private:
  T tensor_;
};

template <typename T>
struct tensor_traits<TensorAccumulator<T>> : public tensor_traits<T> {
};

}

#endif
