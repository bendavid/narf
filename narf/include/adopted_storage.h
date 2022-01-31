#ifndef NARF_ADOPTED_STORAGE_H
#define NARF_ADOPTED_STORAGE_H

#include <boost/histogram.hpp>
#include "view.h"
#include "traits.h"

namespace narf {

  template <typename T, bool do_init = true>
  class adopted_storage {
  public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    adopted_storage(void *buffer, std::size_t buffer_size) :
        size_bytes_(buffer_size),
        bytes_(static_cast<std::byte*>(std::align(alignof(T), 0, buffer, size_bytes_))) {
      // adopting memory in this way is only safe if the relevant classes have standard layout
      static_assert(std::is_standard_layout<T>::value);
      // the destructor will never be called
      static_assert(std::is_trivially_destructible<T>::value);

      if (size_bytes_ != buffer_size || bytes_ == nullptr) {
        throw std::runtime_error("cannot adopt storage if alignment is incorrect");
      }
    }

    void reset(std::size_t n) {
      if (reinterpret_cast<std::byte*>(data_.data()) != bytes_) {
        // array_view initialization
        if constexpr (do_init) {
          // elements are already default initialized in this case
          data_ = array_view<T>(bytes_, size_bytes_, in_place);
        }
        else {
          // special case, re-use the existing values
          data_ = array_view<T>(bytes_, size_bytes_);
        }
      }
      else {
        std::fill_n(data_.begin(), std::min(n, data_.size()), T{});
      }

      if (n > data_.size()) {
        throw std::runtime_error("requested size too large");
      }
      size_ = n;
    }

    std::size_t size() const noexcept { return size_; }

    T *begin() noexcept { return data_.begin(); }
    const T *begin() const noexcept { return data_.begin(); }
    const T *cbegin() const noexcept { return data_.cbegin(); }

    T *end() noexcept { return data_.end(); }
    const T *end() const noexcept { return data_.end(); }
    const T *cend() const noexcept { return data_.cend(); }

    T &operator [](std::size_t i) noexcept { return data_[i]; }
    const T &operator [](std::size_t i) const noexcept { return data_[i]; }

    T &at(std::size_t i) noexcept { return data_.at(i); }
    const T &at(std::size_t i) const noexcept { return data_.at(i); }

    template <class U, class = boost::histogram::detail::requires_iterable<U>>
    bool operator==(const U& u) const {
      using std::begin;
      using std::end;
      using namespace boost::histogram;
      return std::equal(this->begin(), this->end(), begin(u), end(u), detail::safe_equal{});
    }

    static constexpr bool has_threading_support = boost::histogram::accumulators::is_thread_safe<T>::value;

  private:
    std::size_t size_{};

    std::size_t size_bytes_;
    std::byte *bytes_;

    array_view<T> data_;
  };

  template <typename T, bool do_init>
  struct storage_traits<adopted_storage<T, do_init>> {
    static constexpr bool is_adopted = true;
  };

}

#endif
