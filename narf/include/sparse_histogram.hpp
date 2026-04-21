#ifndef NARF_SPARSE_HISTOGRAM_HPP
#define NARF_SPARSE_HISTOGRAM_HPP

#include <atomic>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/histogram.hpp>
#include <boost/histogram/unsafe_access.hpp>

#include "atomic_adaptor.hpp"
#include "concurrent_flat_map.hpp"

namespace narf {

// A boost::histogram-compatible Storage backed by narf::concurrent_flat_map.
//
// Bins are addressed by their linearized index (the same scheme boost
// histogram itself uses for dense storages). Only bins that have been touched
// by a fill consume memory; the underlying lock-free map allows concurrent
// fills from many threads.
//
// Iteration via the standard begin()/end() interface walks every virtual bin
// (including never-touched bins, which materialize on access via the map's
// emplace path). For sparse traversals prefer iterating data() directly with
// for_each.
//
// Conversion of histograms using this storage to python hist.Hist objects is
// not supported in the current implementation.
template <typename T = narf::atomic_adaptor<double>>
class concurrent_sparse_storage {
public:
  using value_type      = T;
  using reference       = T&;
  using const_reference = const T&;
  using map_type        = concurrent_flat_map<std::size_t, T>;

  static constexpr bool has_threading_support = true;

  concurrent_sparse_storage() = default;
  explicit concurrent_sparse_storage(double fill_fraction)
    : fill_fraction_(fill_fraction) {}
  concurrent_sparse_storage(concurrent_sparse_storage&&) = default;
  concurrent_sparse_storage& operator=(concurrent_sparse_storage&&) = default;
  concurrent_sparse_storage(const concurrent_sparse_storage&) = delete;
  concurrent_sparse_storage& operator=(const concurrent_sparse_storage&) = delete;

  // boost::histogram::histogram calls reset() with the total number of bins
  // (including under/overflow) on construction and after axis growth.
  // The map is sized to fill_fraction * n to avoid most early expansions
  // when an estimate of occupancy is supplied by the caller.
  void reset(std::size_t n) {
    size_ = n;
    const double cap_d = fill_fraction_ * static_cast<double>(n);
    std::size_t cap = cap_d > 1.0 ? static_cast<std::size_t>(cap_d) : 1;
    map_ = map_type{cap};
  }

  double fill_fraction() const noexcept { return fill_fraction_; }

  std::size_t size() const noexcept { return size_; }

  // Insert-on-access; safe for concurrent fills.
  reference operator[](std::size_t i) {
    return *map_.emplace(i).first;
  }

  const_reference operator[](std::size_t i) const {
    if (auto* p = map_.find(i)) return *p;
    // Materialize a default-constructed cell so the const overload still
    // returns a stable reference. This matches the dense_storage contract
    // that "every bin index in [0, size()) is addressable".
    return *const_cast<map_type&>(map_).emplace(i).first;
  }

  // Required by boost::histogram::storage concept, but not used by our fill
  // path. Two sparse storages compare equal iff they have the same logical
  // size and identical populated entries.
  bool operator==(const concurrent_sparse_storage& other) const {
    if (size_ != other.size_) return false;
    bool eq = true;
    map_.for_each([&](std::size_t k, const T& v) {
      if (!eq) return;
      auto* p = const_cast<map_type&>(other.map_).find(k);
      if (!p || !(*p == v)) eq = false;
    });
    return eq;
  }

  // Random-access iterator over the full virtual bin range. Dereferencing
  // materializes the bin (via operator[]). boost::histogram's fill path
  // requires random-access semantics so it can do `begin() + idx`.
  class iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = T;
    using reference         = T&;
    using pointer           = T*;
    using difference_type   = std::ptrdiff_t;

    iterator() = default;
    iterator(concurrent_sparse_storage* s, std::size_t i) : s_(s), i_(i) {}

    reference operator*()  const { return (*s_)[i_]; }
    reference operator[](difference_type n) const {
      return (*s_)[i_ + static_cast<std::size_t>(n)];
    }

    iterator& operator++() { ++i_; return *this; }
    iterator  operator++(int) { auto t = *this; ++i_; return t; }
    iterator& operator--() { --i_; return *this; }
    iterator  operator--(int) { auto t = *this; --i_; return t; }

    iterator& operator+=(difference_type n) {
      i_ = static_cast<std::size_t>(static_cast<difference_type>(i_) + n);
      return *this;
    }
    iterator& operator-=(difference_type n) { return *this += -n; }
    iterator  operator+(difference_type n) const { auto t = *this; t += n; return t; }
    iterator  operator-(difference_type n) const { auto t = *this; t -= n; return t; }
    friend iterator operator+(difference_type n, iterator it) { return it + n; }
    difference_type operator-(const iterator& o) const {
      return static_cast<difference_type>(i_) - static_cast<difference_type>(o.i_);
    }

    bool operator==(const iterator& o) const { return i_ == o.i_; }
    bool operator!=(const iterator& o) const { return i_ != o.i_; }
    bool operator< (const iterator& o) const { return i_ <  o.i_; }
    bool operator<=(const iterator& o) const { return i_ <= o.i_; }
    bool operator> (const iterator& o) const { return i_ >  o.i_; }
    bool operator>=(const iterator& o) const { return i_ >= o.i_; }

  private:
    concurrent_sparse_storage* s_ = nullptr;
    std::size_t i_ = 0;
  };
  using const_iterator = iterator;

  iterator begin() { return iterator(this, 0); }
  iterator end()   { return iterator(this, size_); }
  const_iterator begin() const {
    return iterator(const_cast<concurrent_sparse_storage*>(this), 0);
  }
  const_iterator end() const {
    return iterator(const_cast<concurrent_sparse_storage*>(this), size_);
  }

  map_type& data() noexcept { return map_; }
  const map_type& data() const noexcept { return map_; }

private:
  double      fill_fraction_ = 1.0;
  std::size_t size_ = 0;
  map_type    map_;
};

// Convenience factory: builds a boost::histogram::histogram with the
// concurrent sparse storage and the supplied axes.
template <typename T, typename... Axes>
boost::histogram::histogram<std::tuple<std::decay_t<Axes>...>,
                            concurrent_sparse_storage<T>>
make_histogram_sparse(double fill_fraction, Axes&&... axes) {
  return boost::histogram::make_histogram_with(
      concurrent_sparse_storage<T>{fill_fraction},
      std::forward<Axes>(axes)...);
}

// Helpers to inspect the underlying concurrent_flat_map of a sparse-storage
// boost histogram. Free functions because cppyy does not expose
// boost::histogram::histogram::storage_ directly.
template <typename Axes, typename T>
typename concurrent_sparse_storage<T>::map_type&
sparse_histogram_data(
    boost::histogram::histogram<Axes, concurrent_sparse_storage<T>>& h) {
  return boost::histogram::unsafe_access::storage(h).data();
}

// Snapshot the populated bins of a sparse-storage boost histogram into a
// vector of (linearized_index, value) pairs. Convenience for python
// inspection where passing a python callable to for_each is awkward.
template <typename Axes, typename T>
std::vector<std::pair<std::size_t, double>> sparse_histogram_snapshot(
    boost::histogram::histogram<Axes, concurrent_sparse_storage<T>>& h) {
  std::vector<std::pair<std::size_t, double>> out;
  auto& m = boost::histogram::unsafe_access::storage(h).data();
  out.reserve(m.size());
  m.for_each([&](std::size_t k, const T& v) {
    if constexpr (requires { v.load(); }) {
      out.emplace_back(k, static_cast<double>(v.load()));
    } else {
      out.emplace_back(k, static_cast<double>(v));
    }
  });
  return out;
}

} // namespace narf

#endif // NARF_SPARSE_HISTOGRAM_HPP
