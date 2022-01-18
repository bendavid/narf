
#ifndef NARF_TENSORUTILS_H
#define NARF_TENSORUTILS_H

#include "traits.h"

namespace narf {

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
    // make sure rhs has the same shape and storage order, otherwise the linear element-wise
    // operation doesn't make sense
    static_assert(std::is_same_v<typename U::Dimensions, typename T::Dimensions> && U::Options == T::Options);
    const typename U::Scalar *itr = rhs.data();
    for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it, ++itr) {
      (*it) += *itr;
    }
    return *this;
  }

  // don't use T::operator*= because this is not safe for use with atomics
  TensorAccumulator &operator*= (const TensorAccumulator &rhs) {
    const typename T::Scalar *itr = rhs.tensor_.data();
    for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it, ++itr) {
      (*it) *= *itr;
    }
  }

  // don't use T::operator/= because this is not safe for use with atomics
  TensorAccumulator &operator/= (const TensorAccumulator &rhs) {
    const typename T::Scalar *itr = rhs.tensor_.data();
    for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it, ++itr) {
      (*it) /= *itr;
    }
  }

  // multiplication with scalar
  template <typename U>
  TensorAccumulator &operator*= (const U &rhs) {
    for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it) {
      (*it) *= rhs;
    }
  }

  // division with scalar
  template <typename U>
  TensorAccumulator &operator/= (const U &rhs) {
    for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it) {
      (*it) /= rhs;
    }
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
