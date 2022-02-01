#include <cstring>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>
#include <bit>

#ifndef NARF_VIEW_H
#define NARF_VIEW_H

namespace narf {

  template <typename T>
  T *start_lifetime_as(void *p) {
#ifdef __clang__
    // minimal standard-compliant version (which clang is known to handle correctly)
    // Under implicit lifetime rules http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0593r6.html
    // this is interpreted as equivalent to:
    // 1) copying contents of p to an intermediate buffer
    // 2) implicitly creating instance of T in memory pointed to by p
    // 3) copying contents of intermediate buffer back to p
    // 4) getting a pointer to the implicitly created T and returning it
    // In practice clang will (correctly) optimize this to a no-op
    std::memmove(p, p, sizeof(T));
    return std::launder(static_cast<T*>(p));
#else
    // at least gcc is known to not handle the above code properly
    // (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95349)
    // This version can be interpreted as equivalent to
    // 1) copying contents of p to an intermediate buffer
    // 2) implicitly creating array of bytes and its elements in the memory pointed to by p
    // 3) copying contents of intermediate buffer back to p
    // 4) getting a pointer to the implicitly created array of bytes
    // 5) copying contents of array to another intermediate buffer
    // 6) implicitly creating instance of T in memory pointed to by p
    // 7) copying contents of intermediate buffer back to p
    // 8) getting a pointer to the implicitly created T and returning it
    // In practice gcc will optimize out the first memmove, but the presence of std::launder
    // for the source of the second memmove prevents it from being optimized out
    // (and unfortunately this is the only known way to get the correct behaviour,
    // by erasing the dynamic type)
    // n.b. this code is also in principle standard compliant and correct, so it SHOULD be fine
    // on any other standard-compliant compiler, but might be harder to optimize
    std::memmove(p, p, sizeof(T));
    std::byte *b = std::launder(static_cast<std::byte*>(p));
    std::memmove(p, b, sizeof(T));
    return std::launder(static_cast<T*>(p));
#endif

  }

  template <typename T>
  T *start_lifetime_as_array(void *p, std::size_t size) {
    // n.b. std::launder for std::memmove src is not required under the standard, but needed
    // to work around a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95349
    if (p) {
      // see explanation above in start_lifetime_as
      // logic is directly equivalent except the implicitly created object is an array of T
      // and its elements
#ifdef __clang__
      std::cout << "clang path\n";
      std::memmove(p, p, size*sizeof(T));
      return std::launder(static_cast<T*>(p));
#else
      std::cout << "gcc path\n";
      std::memmove(p, p, size*sizeof(T));
      std::byte *b = std::launder(static_cast<std::byte*>(p));
      std::memmove(p, b, size*sizeof(T));
      return std::launder(static_cast<T*>(p));
#endif
    }
    return static_cast<T*>(p);
  }

  // type trait for implicit lifetime type
  template <typename T>
  struct is_implicit_lifetime {
    static constexpr bool value = std::is_scalar_v<T>
                                || std::is_array_v<T>
                                || std::is_aggregate_v<T>
                                || ((std::is_class_v<T> || std::is_union_v<T>)
                                  && (std::is_trivially_default_constructible_v<T>
                                    || std::is_trivially_copy_constructible_v<T>
                                    || std::is_trivially_move_constructible_v<T>)
                                  && std::is_trivially_destructible_v<T>);
  };

  template< class T >
  inline constexpr bool is_implicit_lifetime_v = is_implicit_lifetime<T>::value;


  // like std::bit_cast except it takes a pointer for the src, so that it can easily
  // work with eg an array of bytes
  template <class To, class From>
  std::enable_if_t<
    std::is_trivially_copyable_v<From> &&
    std::is_trivially_copyable_v<To>,
    To>
  // constexpr support needs compiler magic
  bit_cast_ptr(const From *src) noexcept
  {

    static_assert(is_implicit_lifetime_v<To> || std::is_default_constructible_v<To>,
      "This implementation additionally requires destination type to be either implicit lifetime or default constructible.");

    if constexpr (is_implicit_lifetime_v<To>) {
      alignas(To) std::byte storage[sizeof(To)];
      // implicitly creates To
      std::memcpy(storage, src, sizeof(To));
      To &dst = *std::launder(reinterpret_cast<To*>(storage));
      return dst;
    }
    else if constexpr (std::is_default_constructible_v<To>) {
      To dst;
      std::memcpy(&dst, src, sizeof(To));
      return dst;
    }
  }

#if __cplusplus < 202002L
  template <class To, class From>
  std::enable_if_t<
    sizeof(To) == sizeof(From) &&
    std::is_trivially_copyable_v<From> &&
    std::is_trivially_copyable_v<To>,
    To>
  // constexpr support needs compiler magic
  bit_cast(const From& src) noexcept
  {
    return bit_cast_ptr<To>(&src);
  }
#else
  template<class To, class From>
  using bit_cast = std::bit_cast<To, From>;
#endif



  // class which temporarily reinterprets an object as a different type without making a copy.
  // Depending on which constructor is used, the object representation of the initial object
  // is initially preserved, or replaced by initialization of a new object.
  // Object representation is always preserved when the view is destructed

  //TODO check alignment?
  // we could think to check for standard layout here, but bit_cast doesn't do it, so we leave that out for consistency

  template <typename T>
  class view {

  public:

    template <typename U,
              typename = std::enable_if_t<is_implicit_lifetime_v<U> && std::is_trivially_copyable_v<U>
                    && is_implicit_lifetime_v<T> && std::is_trivially_copyable_v<T>
                    && sizeof(U) == sizeof(T)>>
    view(U &from) : data_(start_lifetime_as<T>(&from)) {}

    template <typename U,
              typename = std::enable_if_t<is_implicit_lifetime_v<U> && std::is_trivially_copyable_v<U>
                    && is_implicit_lifetime_v<T> && std::is_trivially_copyable_v<T>
                    && sizeof(T) % sizeof(U) == 0>>
    view(U *from, std::size_t /*from_size*/) : data_(start_lifetime_as<T>(from)) {
      //TODO check size at runtime?
    }

    template <typename U, typename... Args,
              typename  = std::enable_if_t<is_implicit_lifetime_v<U>
                                        && std::is_trivially_copyable_v<U>
                                        && sizeof(U) == sizeof(T)
                                        && std::is_constructible_v<T, Args...>>>
    view(U &from, std::in_place_t, Args&&... args) : data_(new (&from) T(std::forward<Args>(args)...)) {
      // destructor of T need not be trivial, but program must not rely on its side effects
    }

    template <typename U, typename... Args,
              typename = std::enable_if_t<is_implicit_lifetime_v<U>
                                        && std::is_trivially_copyable_v<U>
                                        && sizeof(T) % sizeof(U) == 0
                                        && std::is_constructible_v<T, Args...>>>
    view(U *from, std::size_t /*from_size*/, std::in_place_t, Args&&... args) : data_(new (from) T(std::forward<Args>(args)...)) {
      // destructor of T need not be trivial, but program must not rely on its side effects

      //TODO check size at runtime?
    }

    // can't be copied since underlying storage must be uniquely managed
    view(const view&) = delete;
    view &operator=(const view&) = delete;

    view(view &&other) : data_(other.data_) {
      other.data_ = nullptr;
    }

    view &operator=(view &&other) {
      if (this != &other) {
        restore_original();
        data_ = other.data_;
        other.data_ = nullptr;
      }

      return *this;
    }

    ~view() {
      restore_original();
    }

    T &operator* () { return *data_; }
    const T &operator* () const { return *data_; }

    T *operator->() { return data_; }
    const T *operator->() const { return data_; }

  private:

    void restore_original() {
      // implicitly restore original object via byte array underlying storage
      if (data_) {
        // can be nullptr in case this object was moved from
        start_lifetime_as<std::byte[sizeof(T)]>(data_);
      }
    }

    T *data_;
  };

  // class which temporarily reinterprets an object as an array of objects,
  // possibly of different type without making a copy.
  // Depending on which constructor is used, the object representation of the initial object
  // is initially preserved, or replaced by initialization of a new array and its members.
  // Object representation is always preserved when the array_view is destructed

  // TODO in c++ 20 and later inherit from std::span?
  // TODO in c++ 20 and later take span as argument instead of pointer and size

  template <typename T>
  class array_view {
  public:

    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using reverse_iterator = std::reverse_iterator<iterator>;

    array_view() = default;

    template <typename U,
              typename = std::enable_if_t<is_implicit_lifetime_v<U> && std::is_trivially_copyable_v<U>
                    && is_implicit_lifetime_v<T> && std::is_trivially_copyable_v<T>
                    && sizeof(U) % sizeof(T) == 0>>
    array_view (U &from) : size_(sizeof(U) / sizeof(T)),  data_(start_lifetime_as_array<T>(&from, size_)) {
      static_assert(is_implicit_lifetime_v<U> && std::is_trivially_copyable_v<U>
                    && is_implicit_lifetime_v<T> && std::is_trivially_copyable_v<T>
                    && sizeof(U) % sizeof(T) == 0);
    };

    template <typename U,
              typename = std::enable_if_t<is_implicit_lifetime_v<U>
                                        && std::is_trivially_copyable_v<U>
                                        && is_implicit_lifetime_v<T>
                                        && std::is_trivially_copyable_v<T>>>
    array_view (U *from, std::size_t from_size) : size_(from_size*sizeof(U) / sizeof(T)),  data_(start_lifetime_as_array<T>(from, size_)) {
      //TODO check size at runtime?
    };

    template <typename U, typename... Args,
        typename = std::enable_if_t<is_implicit_lifetime_v<U> &&
                                    std::is_trivially_copyable_v<U> &&
                                    sizeof(U) % sizeof(T) == 0
                                    && std::is_constructible_v<T, Args...>>>
    array_view(U &from, std::in_place_t, Args&&... args) : size_(sizeof(U) / sizeof(T)),
                                                  data_(static_cast<T*>(::operator new[] (size_*sizeof(T), &from))) {

      // array of T is implicitly created by operator new[] and a pointer to a suitable object is produced,
      // but array elements are not (necessarily) created yet

      // destructor of T need not be trivial, but program must not rely on its side effects

      // begin lifetime and initialize T objects
      for (T *it = data_; it != data_ + size_; ++it) {
        new (it) T(std::forward<Args>(args)...);
      }

    }

//     template <typename U, typename... Args>
    template <typename U, typename... Args,
        typename = std::enable_if_t<is_implicit_lifetime_v<U>
                                  && std::is_trivially_copyable_v<U>
                                  && std::is_constructible_v<T, Args...>>>
    array_view(U *from, std::size_t from_size, std::in_place_t, Args&&... args) : size_(from_size * sizeof(U) / sizeof(T)),
                                                  data_(static_cast<T*>(::operator new[] (size_*sizeof(T), from))) {

      // array of T is implicitly created by operator new[] and a pointer to a suitable object is produced,
      // but array elements are not (necessarily) created yet

      // destructor of T need not be trivial, but program must not rely on its side effects

      //TODO check size at runtime?

      // begin lifetime and initialize T objects
      for (T *it = data_; it != data_ + size_; ++it) {
        new (it) T(std::forward<Args>(args)...);
      }

    }

    // can't be copied since underlying storage must be uniquely managed
    array_view(const array_view&) = delete;
    array_view &operator=(const array_view&) = delete;

    array_view(array_view &&other) : size_(other.size_), data_(other.data_) {
      other.size_ = 0;
      other.data_ = nullptr;
    }

    array_view &operator=(array_view &&other) {
      if (this != &other) {
        restore_original();

        size_ = other.size_;
        data_ = other.data_;

        other.size_ = 0;
        other.data_ = nullptr;
      }

      return *this;
    }

    ~array_view() {
      restore_original();
    }

    std::size_t size() const { return size_; }
    std::size_t size_bytes() const { return size_*sizeof(T); }
    bool empty() const { return size_ == 0; }


    reference front() { return *data_[0]; }
    const_reference front() const { return *data_[0]; }

    reference back() { return *data_[size_ - 1]; }
    const_reference back() const { return *data_[size_ - 1]; }

    reference operator[](std::size_t i) { return data_[i]; }
    const_reference operator[](std::size_t i) const { return data_[i]; }

    reference at(std::size_t i) { return data_[i]; }
    const_reference at(std::size_t i) const { return data_[i]; }

    iterator begin() { return data_; }
    const iterator begin() const { return data_; }
    const iterator cbegin() const { return data_; }

    iterator end() { return data_ + size_; }
    const iterator end() const { return data_ + size_; }
    const iterator cend() const { return data_ + size_; }

    reverse_iterator rbegin() { return reverse_iterator(data_ + size_); }
    const reverse_iterator rbegin() const { return reverse_iterator(data_ + size_); }
    const reverse_iterator crbegin() const { return reverse_iterator(data_ + size_); }

    reverse_iterator rend() { return reverse_iterator(data_); }
    const reverse_iterator rend() const { return reverse_iterator(data_); }
    const reverse_iterator crend() const { return reverse_iterator(data_); }

    pointer data() { return data_; }
    const_pointer data() const { return data_; }

  private:

    void restore_original() {
      // implicitly restore original object via byte array underlying storage
      start_lifetime_as_array<std::byte>(data_, size_*sizeof(T));
    }

    std::size_t size_{};
    T *data_{};
  };

}

#endif
