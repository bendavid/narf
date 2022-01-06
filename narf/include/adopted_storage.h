#ifndef NARF_ADOPTED_STORAGE_H
#define NARF_ADOPTED_STORAGE_H

#include <boost/histogram.hpp>

namespace narf {

  template <typename T>
  class adopted_storage {
  public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    adopted_storage(void *buffer, std::size_t buffer_size) :
        buffer_size_(buffer_size),
        data_(static_cast<T*>(std::align(alignof(T), sizeof(T), buffer, buffer_size_))) {
      // adopting memory in this way is only safe if the relevant classes have standard layout
      static_assert(std::is_standard_layout<T>::value);
      // the destructor will never be called
      static_assert(std::is_trivially_destructible<T>::value);
    }

    void reset(std::size_t n) {
      if (n > initialized_size_) {
        if (n > max_size()) {
          throw std::runtime_error("requested size too large");
        }
        data_ = new (data_) T[n]();
        initialized_size_ = n;
      }
      else {
        std::fill_n(data_, n, T());
      }
      size_ = n;
    }

    std::size_t size() const noexcept { return size_; }

    T *begin() noexcept { return data_; }
    const T *begin() const noexcept { return data_; }
    const T *cbegin() const noexcept { return data_; }

    T *end() noexcept { return data_ + size_; }
    const T *end() const noexcept { return data_ + size_; }
    const T *cend() const noexcept { return data_ + size_; }

    T &operator [](std::size_t i) noexcept { return data_[i]; }
    const T &operator [](std::size_t i) const noexcept { return data_[i]; }

    T &at(std::size_t i) noexcept { return data_[i]; }
    const T &at(std::size_t i) const noexcept { return data_[i]; }

    template <class U, class = boost::histogram::detail::requires_iterable<U>>
    bool operator==(const U& u) const {
      using std::begin;
      using std::end;
      using namespace boost::histogram;
      return std::equal(this->begin(), this->end(), begin(u), end(u), detail::safe_equal{});
    }

    static constexpr bool has_threading_support = boost::histogram::accumulators::is_thread_safe<T>::value;

  private:

    std::size_t max_size() const noexcept { return buffer_size_/sizeof(T); }

    std::size_t size_{};
    std::size_t initialized_size_{};

    std::size_t buffer_size_;
    T *data_;
  };

}

#endif
