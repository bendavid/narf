
#ifndef NARF_TENSORUTILS_H
#define NARF_TENSORUTILS_H

#include "traits.h"

namespace narf {

template <typename T>
class TensorAccumulator : public T {

public:

  // constructors from base
//   using T::T;
  TensorAccumulator() { T::setZero(); }

//   TensorAccumulator(const T &other) : T(other) {}
//   TensorAccumulator() = default;

//   template <typename U>
//   TensorAccumulator(const U &other) : T(static_cast<const T&>(other)) {}
//
//   template <typename U>
//   TensorAccumulator(U &&other) : T(static_cast<T&&>(std::move(other))) {}

//   TensorAccumulator(U &&other) : T(static_cast<const T&>(other)) {}

//   TensorAccumulator(const T &other) : T(other) {}
//   TensorAccumulator(T &&other) : T(std::move(other)) {}

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

//   TensorAccumulator &operator+=(const TensorAccumulator &rhs) {
//     T::operator+=(rhs);
//     return *this;
//   }

//   TensorAccumulator &operator+=(const boost::histogram::weight_type<TensorAccumulator> &rhs)
//   {
//     const typename T::Scalar *itr = rhs.value.data();
//     for (typename T::Scalar *it = T::data(); it != T::data() + size(); ++it, ++itr) {
//       (*it) += *itr;
//     }
//     return *this;
//   }
//
  TensorAccumulator &operator+=(const boost::histogram::weight_type<const T&> &rhs)
  {
//     const typename T::Scalar *itr = rhs.value.data();
//     for (typename T::Scalar *it = T::data(); it != T::data() + size(); ++it, ++itr) {
//       (*it) += *itr;
//     }
    T::operator+=(rhs.value());
    return *this;
  }


//   template <typename U>
//   TensorAccumulator &operator+=(const boost::histogram::weight_type<const U&> &rhs)
//   {
//     const typename T::Scalar *itr = rhs.value.data();
//     for (typename T::Scalar *it = T::data(); it != T::data() + size(); ++it, ++itr) {
//       (*it) += *itr;
//     }
//     return *this;
//   }
//
//   // don't use T::operator+= because this is not safe for use with atomics, being implemented as
//   // lhs = lhs + rhs
//   template <typename U>
//   TensorAccumulator &operator+=(const U &rhs) {
//     // make sure rhs has the same shape and storage order, otherwise the linear element-wise
//     // operation doesn't make sense
//     static_assert(std::is_same_v<typename U::Dimensions, typename T::Dimensions> && U::Options == T::Options);
//     const typename U::Scalar *itr = rhs.data();
//     for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it, ++itr) {
//       (*it) += *itr;
//     }
//     return *this;
//   }
//
//   // don't use T::operator*= because this is not safe for use with atomics
//   TensorAccumulator &operator*= (const TensorAccumulator &rhs) {
//     const typename T::Scalar *itr = rhs.tensor_.data();
//     for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it, ++itr) {
//       (*it) *= *itr;
//     }
//   }
//
//   // don't use T::operator/= because this is not safe for use with atomics
//   TensorAccumulator &operator/= (const TensorAccumulator &rhs) {
//     const typename T::Scalar *itr = rhs.tensor_.data();
//     for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it, ++itr) {
//       (*it) /= *itr;
//     }
//   }
//
//   // multiplication with scalar
//   template <typename U>
//   TensorAccumulator &operator*= (const U &rhs) {
//     for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it) {
//       (*it) *= rhs;
//     }
//   }
//
//   // division with scalar
//   template <typename U>
//   TensorAccumulator &operator/= (const U &rhs) {
//     for (typename T::Scalar *it = tensor_.data(); it != tensor_.data() + size(); ++it) {
//       (*it) /= rhs;
//     }
//   }
//
//   template <typename... Us>
//   typename T::Scalar &operator() (Us&&... us) { return tensor_(std::forward<Us>(us)...); }
//
//   template <typename... Us>
//   const typename T::Scalar &operator() (Us&&... us) const { return tensor_(std::forward<Us>(us)...); }
//
// private:
//   T tensor_;
};


// pass-through tensor_traits
template <typename T>
struct tensor_traits<TensorAccumulator<T>> : public tensor_traits<T> {
};


}

namespace boost {
namespace histogram {

  // specializations for weight_type so that underlying tensors are used
//   template <typename T>
//   struct weight_type<narf::TensorAccumulator<T>> : public weight_type<T> {
//
//
// //     weight_type(const T &other) : weight_type<T>{other} { std::cout << "converting tensor constructor\n"; }
//   };

  template <typename T>
  struct weight_type<narf::TensorAccumulator<T>> : public weight_type<const T&> {
    weight_type(const weight_type<T>& other) : weight_type<const T&>(other)  {}
    weight_type(const weight_type<T&>& other) : weight_type<const T&>(other)  { std::cout << "conv1\n"; }
    weight_type(const weight_type<T&&>& other) : weight_type<const T&>(other)  {}
    weight_type(const weight_type<const T>& other) : weight_type<const T&>(other)  {}
    weight_type(const weight_type<const T&>& other) : weight_type<const T&>(other)  {}
    weight_type(const weight_type<const T&&>& other) : weight_type<const T&>(other)  {}
  };

//   template <typename T>
//   struct weight_type<narf::TensorAccumulator<T>&> : public weight_type<T&> {
// //     weight_type() {}
// //     weight_type(const weight_type<T&>& other) : weight_type<T&>(other) { std::cout << "converting tensor constructor lvalref\n"; }
//   };
//
//   template <typename T>
//   struct weight_type<narf::TensorAccumulator<T>&&> : public weight_type<T&&> {
// //     weight_type() {}
//   };
//
//   template <typename T>
//   struct weight_type<const narf::TensorAccumulator<T>> : public weight_type<const T> {
// //     weight_type() {}
//   };
//
//   template <typename T>
//   struct weight_type<const narf::TensorAccumulator<T>&> : public weight_type<const T&> {
// //     weight_type() {}
// //     weight_type(const weight_type<const T&>& other) : weight_type<const T&>(other) { std::cout << "converting tensor constructor const lvalref\n"; }
//   };
//
//   template <typename T>
//   struct weight_type<const narf::TensorAccumulator<T>&&> : public weight_type<const T&&> {
// //     weight_type() {}
//   };

}
}

// namespace Eigen {
// namespace internal {
//
//   template <typename T>
//   struct traits<narf::TensorAccumulator<T>> : public traits<T> {
//   };
//
// }
// }

#endif
