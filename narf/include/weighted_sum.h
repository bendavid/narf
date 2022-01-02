#ifndef NARF_WEIGHTED_SUM_H
#define NARF_WEIGHTED_SUM_H

#include <boost/histogram/detail/atomic_number.hpp>
#include <boost/core/nvp.hpp>
#include <boost/histogram/fwd.hpp> // for weighted_sum<>
#include <type_traits>
#include <iostream>

namespace narf {

  using namespace boost::histogram;

  /// Holds sum of weights and its variance estimate
  template <class ValueType, bool ThreadSafe>
  class weighted_sum {
    using internal_type =
        std::conditional_t<ThreadSafe,detail::atomic_number<ValueType>, ValueType>;

  public:
    using value_type = ValueType;
    using const_reference = const value_type&;

    weighted_sum() = default;

    /// Initialize sum to value and allow implicit conversion
    weighted_sum(const_reference value) noexcept : weighted_sum(value, value) {}

    /// Allow implicit conversion from sum<T>
    template <class T>
    weighted_sum(const weighted_sum<T, ThreadSafe>& s) noexcept
        : weighted_sum(s.value(), s.variance()) {}

    /// Initialize sum to value and variance
    weighted_sum(const_reference value, const_reference variance) noexcept
        : sum_of_weights_(value), sum_of_weights_squared_(variance) {}

    /// Increment by one.
    weighted_sum& operator++() {
      ++sum_of_weights_;
      ++sum_of_weights_squared_;
//       std::cout << "increment 0" << std::endl;
      return *this;
    }

    /// Increment by weight.
    weighted_sum& operator+=(const weight_type<value_type>& w) {
      sum_of_weights_ += w.value;
      sum_of_weights_squared_ += w.value * w.value;
//       std::cout << "increment 1: sum of weights = " << sum_of_weights_.load() <<  " value = " << w.value << std::endl;
      return *this;
    }

    /// Added another weighted sum.
    weighted_sum& operator+=(const weighted_sum& rhs) {
      sum_of_weights_ += rhs.sum_of_weights_;
      sum_of_weights_squared_ += rhs.sum_of_weights_squared_;
//       std::cout << "increment 2" << std::endl;
//       std::cout << "increment 2: sum of weights = " << sum_of_weights_.load() <<  " rhs sum of weights = " << rhs.sum_of_weights_.load() << std::endl;

      return *this;
    }

    /// Scale by value.
    weighted_sum& operator*=(const_reference x) {
      sum_of_weights_ *= x;
      sum_of_weights_squared_ *= x * x;
      return *this;
    }

    bool operator==(const weighted_sum& rhs) const noexcept {
      return sum_of_weights_ == rhs.sum_of_weights_ &&
            sum_of_weights_squared_ == rhs.sum_of_weights_squared_;
    }

    bool operator!=(const weighted_sum& rhs) const noexcept { return !operator==(rhs); }

    /// Return value of the sum.
    value_type value() const noexcept { return sum_of_weights_; }

    /// Return estimated variance of the sum.
    value_type variance() const noexcept { return sum_of_weights_squared_; }

    // lossy conversion must be explicit
    explicit operator value_type() const { return sum_of_weights_; }

    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
      ar& make_nvp("sum_of_weights", sum_of_weights_);
      ar& make_nvp("sum_of_weights_squared", sum_of_weights_squared_);
    }

    static constexpr bool thread_safe() noexcept { return ThreadSafe; }

  private:
    internal_type sum_of_weights_{};
    internal_type sum_of_weights_squared_{};
  };

} // namespace narf


#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED
namespace std {
template <class T, class U, bool B1, bool B2>
struct common_type<narf::weighted_sum<T, B1>,
                   narf::weighted_sum<U, B2>> {
  using type = narf::weighted_sum<common_type_t<T, U>, (B1 || B2)>;
};

template <class T, class U, bool B1>
struct common_type<narf::weighted_sum<T, B1>, U> {
  using type = narf::weighted_sum<common_type_t<T, U>, B1>;
};

template <class T, class U, bool B2>
struct common_type<T, narf::weighted_sum<U, B2>> {
  using type = narf::weighted_sum<common_type_t<T, U>, B2>;
};
} // namespace std
#endif

#endif
