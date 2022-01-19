#ifndef NARF_ATOMIC_ADAPTOR_H
#define NARF_ATOMIC_ADAPTOR_H

#include <atomic>
#include <boost/histogram/detail/priority.hpp>
#include <type_traits>
#include "traits.h"
#include "tensorutils.h"

namespace narf {

template <class T>
struct atomic_adaptor_base : std::atomic<T> {

  // prior to C++20 default constructor for std::atomic is trivial and the resulting object
  // cannot be used without calling std::atomic_init according to the standard
#if __cplusplus < 202002L
  atomic_adaptor_base() noexcept : std::atomic<T>{T()} {}
#else
  atomic_adaptor_base() = default;
#endif

  atomic_adaptor_base(const atomic_adaptor_base& o) noexcept : std::atomic<T>{o.load()} {}
  atomic_adaptor_base& operator=(const atomic_adaptor_base& o) noexcept {
    this->store(o.load());
    return *this;
  }

  atomic_adaptor_base(const T& o) noexcept : std::atomic<T>{o} {}

  static constexpr bool thread_safe() noexcept { return true; }

  // general atomic modification using compare and exchange loop
  template<typename L, typename...Us>
  void modify(L &modifier, Us&&... xs) {
    T expected = this->load(std::memory_order_relaxed);
    T desired = expected;
    modifier(desired, std::forward<Us>(xs)...);
    while (!this->compare_exchange_weak(expected, desired, std::memory_order_seq_cst, std::memory_order_relaxed)) {
      desired = expected;
      modifier(desired, std::forward<Us>(xs)...);
    }
  }

};

// base template for accumulators

template <class T, class Enable = void>
struct atomic_adaptor : atomic_adaptor_base<T> {

private:
#if __cplusplus < 201703L
  template< class... >
  using void_t = void;
#else
  template< class...Ts >
  using void_t = std::void_t<Ts...>;
#endif

public:

  atomic_adaptor() = default;
  atomic_adaptor(const T& o) noexcept : atomic_adaptor_base<T>{o} {}

  // only enable if ++ operator is available for T
  template <typename U = T, typename E = void_t<decltype(++std::declval<U&>())>>
  atomic_adaptor& operator++() noexcept {
    auto modifier = [](T &t) {
      ++t;
    };
    this->modify(modifier);
    return *this;
  }

  // only enable if += operator is available for T with corresponding argument type
  template <typename U, typename E = void_t<decltype(std::declval<T&>() += std::forward<U>(std::declval<U>()))>>
  atomic_adaptor& operator+=(U&& x) noexcept {
    auto modifier = [](T &t, auto &&u) {
      t += std::forward<U>(u);
    };
    this->modify(modifier, std::forward<U>(x));
    return *this;
  }

  // only enable if call operator is available for T with corresponding argument types
  template <typename...Us, typename E = void_t<decltype(std::declval<T&>()(std::forward<Us>(std::declval<Us>())...))>>
  void operator() (Us&&... xs) noexcept {
    auto modifier = [](T &t, auto&&... us) {
      t(std::forward<Us>(us)...);
    };
    this->modify(modifier, std::forward<Us>(xs)...);
  }

  // only enable if *= operator is available for T with corresponding argument type
  template <typename U, typename E = void_t<decltype(std::declval<T&>() *= std::forward<U>(std::declval<U>()))>>
  atomic_adaptor& operator*=(U&& x) noexcept {
    auto modifier = [](T &t, auto &&u) {
      t *= std::forward<U>(u);
    };
    this->modify(modifier, std::forward<U>(x));
    return *this;
  }

  // only enable if /= operator is available for T with corresponding argument type
  template <typename U, typename E = void_t<decltype(std::declval<T&>() /= std::forward<U>(std::declval<U>()))>>
  atomic_adaptor& operator/=(U&& x) noexcept {
    auto modifier = [](T &t, auto &&u) {
      t /= std::forward<U>(u);
    };
    this->modify(modifier, std::forward<U>(x));
    return *this;
  }

};

// specialization for integral types
template <class T>
struct atomic_adaptor<T, std::enable_if_t<std::is_integral<T>::value>> : atomic_adaptor_base<T> {

  atomic_adaptor() = default;
  atomic_adaptor(T o) noexcept : atomic_adaptor_base<T>{o} {}

  atomic_adaptor& operator*=(double x) noexcept {
    auto modifier = [](T &t, double u) {
      t *= u;
    };
    this->modify(modifier, x);
    return *this;
  }

  atomic_adaptor& operator/=(double x) noexcept {
    auto modifier = [](T &t, double u) {
      t /= u;
    };
    this->modify(modifier, x);
    return *this;
  }

  // everything else we need is already provided by atomic_adaptor_base<T> or std::atomic<T>

};

// specialization for floating point types
template <class T>
struct atomic_adaptor<T, std::enable_if_t<std::is_floating_point<T>::value>> : atomic_adaptor_base<T> {

  atomic_adaptor() = default;
  atomic_adaptor(T o) noexcept : atomic_adaptor_base<T>{o} {}

  atomic_adaptor& operator++() noexcept {
    auto modifier = [](T &t) {
      ++t;
    };
    this->modify(modifier);
    return *this;
  }

  // already provided by std::atomic<T> in C++20
#if __cplusplus < 202002L
  atomic_adaptor& operator+=(T x) noexcept {
    auto modifier = [](T &t, T u) {
      t += u;
    };
    this->modify(modifier, x);
    return *this;
  }
#endif

  atomic_adaptor& operator*=(T x) noexcept {
    auto modifier = [](T &t, T u) {
      t *= u;
    };
    this->modify(modifier, x);
    return *this;
  }

  atomic_adaptor& operator/=(T x) noexcept {
    auto modifier = [](T &t, T u) {
      t /= u;
    };
    this->modify(modifier, x);
    return *this;
  }

};


// specialization for tensor case, to move the atomic wrapper inside the tensor (ie atomicity applies
// only element by element)

template <typename T, typename Dimensions_, int Options_, typename IndexType>
struct atomic_adaptor<TensorAccumulator<Eigen::TensorFixedSize<T, Dimensions_, Options_, IndexType>>> : public TensorAccumulator<Eigen::TensorFixedSize<atomic_adaptor<T>, Dimensions_, Options_, IndexType>> {

private:
  using base_t = TensorAccumulator<Eigen::TensorFixedSize<atomic_adaptor<T>, Dimensions_, Options_, IndexType>>;
  using Tensor_t = Eigen::TensorFixedSize<T, Dimensions_, Options_, IndexType>;

public:

  // constructors from base class
  using base_t::base_t;

  // re-implement += operator because Eigen implementation uses lhs = lhs + rhs and
  // does not preserve atomicity
  atomic_adaptor &operator+=(const Tensor_t &rhs) {
    const T *itr = rhs.data();
    for (typename base_t::Scalar *it = base_t::data(); it != base_t::data() + base_t::size(); ++it, ++itr) {
      (*it) += *itr;
    }
    return *this;
  }


  //TODO do we need this generic version?
//   template <typename U>
//   atomic_adaptor &operator+=(const Eigen::TensorFixedSize<U, Dimensions_, Options_, IndexType> &rhs) {
//     const U *itr = rhs.data();
//     for (typename base_t::Scalar *it = base_t::data(); it != base_t::data() + base_t::size(); ++it, ++itr) {
//       (*it) += *itr;
//     }
//     return *this;
//   }

//   atomic_adaptor &operator+=(atomic_adaptor &rhs) {
//     operator+=(rhs);
//     return *this;
//   }

  // avoid unnecessary conversion to atomic and back by taking a weight of the
  // underlying type directly
  atomic_adaptor &operator+=(const boost::histogram::weight_type<const Tensor_t&> &w) {
    operator+=(w.value);
    return *this;
  }

  static constexpr bool thread_safe() noexcept { return true; }
};

// specializations for weighted_sum exploiting the fact that value and variance
// can be incremented independently
// n.b. this leads to slightly weaker atomicity guarantees, in that a
// thread reading a bin concurrent with another writing may
// observe inconsistent value vs variance (e.g. value before modification and variance
// after modification, or vice-versa)
// template <typename T>
// struct atomic_adaptor<boost::histogram::accumulators::weighted_sum<T>> : public boost::histogram::accumulators::weighted_sum<atomic_adaptor<T>> {
//   // constructors from base class
//   using boost::histogram::accumulators::weighted_sum<atomic_adaptor<T>>::weighted_sum;
//
// //   // avoid unnecessary conversion to atomic and back by taking a weight of the
// //   // underlying type directly
// //   atomic_adaptor& operator+=(const boost::histogram::weight_type<const T&>& w) {
// // //   atomic_adaptor& operator+=(const boost::histogram::weight_type<T>& w) {
// //     // FIXME boost::histogram::weighted_sum data members should be protected
// //     // rather than private to avoid the need for const_cast here
// //     const_cast<atomic_adaptor<T>&>(this->value()) += w.value;
// //     // helps for types which use expression templates (e.g. eigen tensors)
// //     const_cast<atomic_adaptor<T>&>(this->variance()) += w.value*w.value;
// //     return *this;
// //   }
//
//   static constexpr bool thread_safe() noexcept { return true; }
//
// };

// template <typename T, typename Dimensions_, int Options_, typename IndexType>
// struct atomic_adaptor<boost::histogram::accumulators::weighted_sum<TensorAccumulator<Eigen::TensorFixedSize<T, Dimensions_, Options_, IndexType>>>> : public boost::histogram::accumulators::weighted_sum<TensorAccumulator<Eigen::TensorFixedSize<atomic_adaptor<T>, Dimensions_, Options_, IndexType>>> {
//
// private:
//   using Tensor_t = Eigen::TensorFixedSize<T, Dimensions_, Options_, IndexType>;
//   using AtomicTensorAcc_t = TensorAccumulator<Eigen::TensorFixedSize<atomic_adaptor<T>, Dimensions_, Options_, IndexType>>;
//   using base_t = boost::histogram::accumulators::weighted_sum<AtomicTensorAcc_t>;
//
// public:
//
//   // constructors from base class
//   using base_t::base_t;
//
//   // avoid unnecessary conversion to atomic and back by taking a weight of the
//   // underlying type directly
//   atomic_adaptor& operator+=(const boost::histogram::weight_type<const Tensor_t&>& w) {
//     // FIXME boost::histogram::weighted_sum data members should be protected
//     // rather than private to avoid the need for const_cast here
//     const_cast<AtomicTensorAcc_t&>(this->value()) += w.value;
//     const Tensor_t wsq = w.value.square();
//     const_cast<AtomicTensorAcc_t&>(this->variance()) += wsq;
//     return *this;
//   }
//
//   static constexpr bool thread_safe() noexcept { return true; }
// };
//
// template <typename T>
// struct acc_traits<atomic_adaptor<T>> : public acc_traits<T> {
// };


} // namespace narf

namespace boost {
namespace histogram {
namespace detail {

// pass-through type traits to underlying type
template <class T>
auto accumulator_traits_impl(narf::atomic_adaptor<T>&, priority<2>)
  -> decltype(accumulator_traits_impl_call_op(&T::operator()));

}

// specializations for weight_type so that underlying types are used
// template <typename T>
// struct weight_type<atomic_adaptor


// template <typename T>
// struct weight_type<narf::atomic_adaptor<T>> : public weight_type<T> {
// private:
//   using base_t = weight_type<T>;
//
// public:
//   weight_type(const base_t &other) : base_t(other) { std::cout << "converting constructor\n"; }
// };

// template <typename T>
// struct weight_type<narf::atomic_adaptor<T>&> : public weight_type<T&> {
// };
//
// template <typename T>
// struct weight_type<narf::atomic_adaptor<T>&&> : public weight_type<T&&> {
// };
//
// template <typename T>
// struct weight_type<const narf::atomic_adaptor<T>> : public weight_type<const T> {
// };
//
// template <typename T>
// struct weight_type<const narf::atomic_adaptor<T>&> : public weight_type<const T&> {
// };
//
// template <typename T>
// struct weight_type<const narf::atomic_adaptor<T>&&> : public weight_type<const T&&> {
// };

}
}

namespace narf {

template <typename T>
struct atomic_adaptor<boost::histogram::accumulators::weighted_sum<T>> : public boost::histogram::accumulators::weighted_sum<atomic_adaptor<T>> {
  // constructors from base class
  using boost::histogram::accumulators::weighted_sum<atomic_adaptor<T>>::weighted_sum;

  // avoid unnecessary conversion to atomic and back by taking a weight of the
  // underlying type directly
//   atomic_adaptor& operator+=(const boost::histogram::weight_type<const T&>& w) {
//   atomic_adaptor& operator+=(const boost::histogram::weight_type<atomic_adaptor<T>>& w) {
//     // FIXME boost::histogram::weighted_sum data members should be protected
//     // rather than private to avoid the need for const_cast here
// //     const_cast<atomic_adaptor<T>&>(this->value()) += w.value;
//     // helps for types which use expression templates (e.g. eigen tensors)
// //     const_cast<atomic_adaptor<T>&>(this->variance()) += w.value*w.value;
//     return *this;
//   }

  static constexpr bool thread_safe() noexcept { return true; }

};

}

#endif
