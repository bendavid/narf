
#ifndef NARF_TENSORUTILS_H
#define NARF_TENSORUTILS_H

#include "traits.h"
#include <boost/histogram.hpp>

namespace narf {

template <typename T>
class TensorAccumulator : public T {

public:
  using tensor_t = T;

  // eigen tensors are otherwise uninitialized
  TensorAccumulator() { T::setZero(); }

  // compile-time size
  static constexpr std::ptrdiff_t size() { return tensor_traits<T>::size; }

  // prefix increment
  TensorAccumulator &operator++()
  {
    for (typename T::Scalar *it = T::data(); it != T::data() + size(); ++it) {
      ++(*it);
    }
    return *this;
  }

  TensorAccumulator &operator+=(const T &rhs) {
    T::operator+=(rhs);
    return *this;
  }

  template <typename U>
  TensorAccumulator &operator+=(const boost::histogram::weight_type<U> &rhs)
  {
    T::operator+=(rhs.value);
    return *this;
  }

};

// pass-through tensor_traits
template <typename T>
struct tensor_traits<TensorAccumulator<T>> : public tensor_traits<T> {
};


// TODO sort out weight_type mess and avoid this
template <typename T>
class tensor_weighted_sum : public boost::histogram::accumulators::weighted_sum<T> {

private:
  using base_t = boost::histogram::accumulators::weighted_sum<T>;

public:

  // constructors from base
  using base_t::base_t;

  template <typename U>
  tensor_weighted_sum& operator+=(const typename boost::histogram::weight_type<U>& w) {
    // FIXME boost::histogram::weighted_sum data members should be protected
    // rather than private to avoid the need for const_cast here
    const_cast<T&>(this->value()) += w.value;
    const_cast<T&>(this->variance()) += w.value*w.value;
    return *this;
  }
};

template <typename T>
struct acc_traits<tensor_weighted_sum<T>> {
  static constexpr bool is_weighted_sum = true;
  using value_type = T;
};

}


#endif
