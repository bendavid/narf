#ifndef NARF_HISTUTILS_H
#define NARF_HISTUTILS_H

#include <boost/histogram.hpp>
#include "weighted_sum.h"
#include "adopted_storage.h"
#include <ROOT/RResultPtr.hxx>
#include <iostream>

namespace narf {
  using namespace boost::histogram;

  template <typename T, typename A>
  adopted_storage<T> make_adopted_storage(A addr, std::size_t size_bytes) {
    void *buffer = reinterpret_cast<void*>(addr);
    return adopted_storage<T>(buffer, size_bytes);
  }

  template<typename Axis, typename... Axes>
  histogram<std::tuple<std::decay_t<Axis>, std::decay_t<Axes>...>, default_storage>
  make_histogram(Axis&& axis, Axes&&... axes) {
    return boost::histogram::make_histogram(std::forward<Axis>(axis), std::forward<Axes>(axes)...);
  }

  template<typename Axis, typename... Axes>
  histogram<std::tuple<std::decay_t<Axis>, std::decay_t<Axes>...>, dense_storage<accumulators::count<double, true>>>
  make_atomic_histogram(Axis&& axis, Axes&&... axes) {
    return make_histogram_with(dense_storage<accumulators::count<double, true>>(), std::forward<Axis>(axis), std::forward<Axes>(axes)...);
  }

  template<typename Axis, typename... Axes>
  histogram<std::tuple<std::decay_t<Axis>, std::decay_t<Axes>...>, dense_storage<boost::histogram::accumulators::weighted_sum<double>>>
  make_histogram_with_error(Axis&& axis, Axes&&... axes) {
    return make_histogram_with(dense_storage<boost::histogram::accumulators::weighted_sum<double>>(), std::forward<Axis>(axis), std::forward<Axes>(axes)...);
  }

  template<typename Axis, typename... Axes>
  histogram<std::tuple<std::decay_t<Axis>, std::decay_t<Axes>...>, dense_storage<narf::weighted_sum<double, true>>>
  make_atomic_histogram_with_error(Axis&& axis, Axes&&... axes) {
    return make_histogram_with(dense_storage<narf::weighted_sum<double, true>>(), std::forward<Axis>(axis), std::forward<Axes>(axes)...);
  }

  template<typename Storage, typename... Axes>
  histogram<std::tuple<std::decay_t<Axes>...>, Storage>
  make_histogram_with_storage(Storage &&storage, Axes&&... axes) {
    return make_histogram_with(std::forward<Storage>(storage), std::forward<Axes>(axes)...);
  }

  template<typename T, typename A, typename... Axes>
  histogram<std::tuple<std::decay_t<Axes>...>, adopted_storage<T>>
  make_histogram_adopted(A addr, std::size_t buf_size, Axes&&... axes) {
    void *buffer = reinterpret_cast<void*>(addr);
    adopted_storage<T> storage(buffer, buf_size);
    auto h = make_histogram_with(std::move(storage), std::forward<Axes>(axes)...);

    if (h.size()*sizeof(T) != buf_size) {
      throw std::runtime_error("size mismatch");
    }

    return h;
  }

  template<typename DFType, typename Helper, typename... ColTypes>
  ROOT::RDF::RResultPtr<typename std::decay_t<Helper>::Result_t>
  book_helper(DFType &df, Helper &&helper, const std::vector<std::string> &colnames) {
    return df.template Book<ColTypes...>(std::forward<Helper>(helper), colnames);
  }

  template<bool underflow, bool overflow, bool circular, bool growth>
  auto get_option() {

    using cond_underflow_t = std::conditional_t<underflow, axis::option::underflow_t, axis::option::none_t>;
    using cond_overflow_t = std::conditional_t<overflow, axis::option::overflow_t, axis::option::none_t>;
    using cond_circular_t = std::conditional_t<circular, axis::option::circular_t, axis::option::none_t>;
    using cond_growth_t = std::conditional_t<growth, axis::option::growth_t, axis::option::none_t>;


    return cond_underflow_t{} | cond_overflow_t{} | cond_circular_t{} | cond_growth_t{};

  }

}


#endif
