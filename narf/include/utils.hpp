#pragma once

#include <ranges>
		
namespace narf {

  template<typename T>
  void print_type_of() {
    std::cout << std::source_location::current().function_name() << std::endl;
  }

  template<typename T>
  void print_type_of(T&& x) {
    print_type_of<decltype(std::forward<T>(x))>();
  }

  // adapts range which is not default constructible for compatibility with RDataFrame
  // Define for example
  // Note that access is unchecked because RDataFrame will never try to iterate before
  // the corresponding Define has executed
  template <typename T>
  class default_range_adapter {
  public:

    default_range_adapter() = default;
    default_range_adapter(const T& range) : range_(range) {}
    default_range_adapter(T&& range) : range_(std::move(range)) {}

    auto begin() { return range_->begin(); }
    auto end() { return range_->end(); }

  private:
    std::optional<T> range_;
  };

  // uses the default_range_adapter if the range is not default constructible, otherwise passthrough
  template <typename T>
  decltype(auto) make_range_with_default(T &&range) {
    if constexpr (std::is_default_constructible_v<T>) {
      return std::forward<T>(range);
    }
    else {
      return default_range_adapter<std::decay_t<T>>(std::forward<T>(range));
    }
  }
  
  // TODO this can eventually be replaced with zip_view in C++23

  template<typename... Ranges>
  class zip_sentinel {
  public:
    using sentinel_t = decltype(std::make_tuple(std::declval<Ranges>().end()...));

    zip_sentinel() = default;
    zip_sentinel(Ranges&... xs) : sentinels_(xs.end()...) {}

    const sentinel_t &sentinels() const { return sentinels_; }

  private:
    sentinel_t sentinels_;
  };

  template <typename... Ranges>
  class zip_iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using iterator_t = decltype(std::make_tuple(std::declval<Ranges>().begin()...));
    using tuple_t = std::tuple<std::ranges::range_reference_t<Ranges>...>;

    zip_iterator(Ranges&... xs) : its_(xs.begin()...) {}

    zip_iterator& operator++() {
      auto op = [](auto&... its){ (++its, ...); };
      std::apply(op, its_);
      return *this;
    }

    zip_iterator operator++(int) {
      auto pre = *this;
      ++*this;
      return pre;
    }

    bool operator==(const zip_sentinel<Ranges...> &sentinel) const {
      auto op = [this, &sentinel]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ( (std::get<Is>(its_) == std::get<Is>(sentinel.sentinels())) || ...);
      };
      return op(std::index_sequence_for<Ranges...>{});
    }

    tuple_t operator*() const {;
      auto op = [](auto&... its)
      { return tuple_t(*its...); };
      auto res = std::apply(op, its_);
      return res;
    }

  private:
    iterator_t its_;
  };

  template<typename... Ts>
  auto make_zip_view(Ts&&... xs) {
    // TODO somehow allow this to inherit sized_iterator behaviour from input ranges?

    auto res = std::views::iota(zip_iterator(xs...), zip_sentinel(xs...))
                  | std::views::transform([](auto const &it){ return *it; });

    return res;
  }

  // TODO this can eventually be replaced with repeat_view in C++23

  template <typename T>
  class repeat_iterator {
  public:
    using difference_type = std::ptrdiff_t;

    repeat_iterator(const T& u) : val_(u) {}

    repeat_iterator& operator++() { return *this; }

    repeat_iterator operator++(int) { return *this; }

    T operator*() const {
      return val_;
    }

  private:
    T val_;
  };

  template<typename T>
  auto make_repeat_view(const T& x) {
    return std::views::iota(repeat_iterator(x))
            | std::views::transform([](auto const &it){ return *it; });
  }

  template <template <typename> class C=ROOT::VecOps::RVec, class T>
  auto range_to(T &&range) {
    using value_type = std::decay_t<decltype(*range.begin())>;

    C<value_type> res;
    if constexpr (std::ranges::sized_range<T>) {
      res.reserve(range.size());
    }
    for (auto const &elem : range) {
      res.emplace_back(elem);
    }

    return res;
  }

  template<typename T>
  static constexpr bool is_container = std::ranges::range<T> && !std::is_same_v<std::decay_t<T>, std::string> && !std::is_same_v<std::decay_t<T>, char*> && !std::is_same_v<std::decay_t<T>, const char*>;


  template <typename T>
  auto make_view(T&& x) {
    if constexpr (is_container<T>) {
      return std::views::all(std::forward<T>(x));
    }
    else {
      return make_repeat_view(std::ref(x))
              | std::views::transform([](auto const &refwrap){ return refwrap.get(); });
    }
  }

} //namespace narf
