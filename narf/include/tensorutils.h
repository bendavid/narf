
#ifndef NARF_TENSORUTILS_H
#define NARF_TENSORUTILS_H

#include "traits.h"
#include <boost/histogram.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace narf {

  using boost::histogram::weight;
  using boost::histogram::weight_type;
  using boost::histogram::detail::priority;


template <typename T, typename Dimensions_>
class tensor_accumulator {

private:

  using tensor_t = Eigen::TensorFixedSize<T, Dimensions_>;

  template <typename U>
  using weight_t = weight_type<const Eigen::TensorFixedSize<U, Dimensions_>&>;

  tensor_t data_;

public:

  // compile-time size
  static constexpr std::ptrdiff_t size = tensor_traits<tensor_t>::size;
  static constexpr std::size_t rank = tensor_traits<tensor_t>::rank;


  // eigen tensors are otherwise uninitialized
  tensor_accumulator() { data_.setZero(); }

  // prefix increment
  tensor_accumulator &operator++()
  {
    for (T *it = data_.data(); it != data_.data() + size; ++it) {
      ++(*it);
    }
    return *this;
  }

  tensor_accumulator &operator+=(const tensor_accumulator &rhs) {
    const T *itr = rhs.data_.data();
    for (T *it = data_.data(); it != data_.data() + size; ++it, ++itr) {
      *it += *itr;
    }
    return *this;
  }

  template <typename U>
  auto increment_impl(const weight_t<U> &rhs, priority<1>) -> decltype(*data_.data() += weight(*rhs.value.data()), *this) {
    const U *itr = rhs.value.data();
    for (T *it = data_.data(); it != data_.data() + size; ++it, ++itr) {
      *it += weight(*itr);
    }
    return *this;
  }

  template <typename U>
  tensor_accumulator &increment_impl(const weight_t<U> &rhs, priority<0>) {
    const U *itr = rhs.value.data();
    for (T *it = data_.data(); it != data_.data() + size; ++it, ++itr) {
      *it += *itr;
    }
    return *this;
  }

  template <typename U>
  tensor_accumulator &operator+=(const weight_t<U> &rhs) {
    return increment_impl(rhs, priority<1>{});
  }

  template <typename... Us>
  auto operator() (Us&&... xs) -> decltype((*data_.data())(std::forward<Us>(xs)...), void()) {
    for (T *it = data_.data(); it != data_.data() + size; ++it) {
      (*it)(std::forward<Us>(xs)...);
    }
  }

  template <typename U, typename... Us>
  auto operator() (const weight_t<U> &w, Us&&... xs) -> decltype((*data_.data())(weight(*w.value.data()), std::forward<Us>(xs)...), void()) {
    const U *itr = w.value.data();
    for (T *it = data_.data(); it != data_.data() + size; ++it, ++itr) {
      (*it)(weight(*itr), std::forward<Us>(xs)...);
    }
  }

  template <typename U>
  tensor_accumulator &operator*=(const U &rhs) {
    for (T *it = data_.data(); it != data_.data() + size; ++it) {
      *it *= rhs;
    }
    return *this;
  }

  tensor_t &data() { return data_; }
  const tensor_t &data() const { return data_; }

};

// pass-through accumulator traits
template <typename T, typename Dimensions_>
struct acc_traits<tensor_accumulator<T, Dimensions_>> {
  static constexpr bool is_weighted_sum = acc_traits<T>::is_weighted_sum;
  static constexpr bool is_tensor = true;
//   using value_type = T;
};

}

namespace boost {
namespace histogram {
namespace detail {

// pass-through accumulator traits to underlying type
template <class T, class Dimensions_>
auto accumulator_traits_impl(narf::tensor_accumulator<T, Dimensions_>&, priority<2>) ->  decltype(accumulator_traits_impl(std::declval<T&>(), priority<2>{}));

}

}
}


#endif
