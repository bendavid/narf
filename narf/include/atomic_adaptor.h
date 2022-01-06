#ifndef NARF_ATOMIC_ADAPTOR_H
#define NARF_ATOMIC_ADAPTOR_H

#include <atomic>
#include <boost/histogram/detail/priority.hpp>
#include <type_traits>

namespace narf {

template <class T>
struct atomic_adaptor_base : std::atomic<T> {

  // prior to C++20 default constructor for std::atomic is trivial and the resulting object
  // cannot be used without calling std::atomic_init according to the standard
#if __cplusplus < 202002L
  atomic_adaptor_base() noexcept : std::atomic<T>{T()} {}
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
    auto modifier = [](T &t, auto&&... xs) {
      t(std::forward<Us>(xs)...);
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


} // namespace narf

namespace boost {
namespace histogram {
namespace detail {

  template <class T>
  auto accumulator_traits_impl(narf::atomic_adaptor<T>&, priority<2>)
    -> decltype(accumulator_traits_impl_call_op(&T::operator()));

}

namespace accumulators {

  template <typename T>
  struct is_thread_safe<weighted_sum<narf::atomic_adaptor<T>>> : std::true_type {};

}

}
}

#endif
