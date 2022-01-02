#ifndef NARF_HISTUTILS_H
#define NARF_HISTUTILS_H

#include <boost/histogram.hpp>
#include "weighted_sum.h"
#include <ROOT/RResultPtr.hxx>
#include <iostream>

namespace narf {
  using namespace boost::histogram;


  // wrapper class to make boost happy by providing dummy allocator_type
  template <typename T>
  class RVecDerived : public ROOT::VecOps::RVec<T> {
  public:
    using allocator_type = std::allocator<T>;

    RVecDerived(T *data, std::size_t size) : ROOT::VecOps::RVec<T>(data, size) {}
  };

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

  template<typename... Axes>
  histogram<std::tuple<std::decay_t<Axes>...>, storage_adaptor<RVecDerived<narf::weighted_sum<double, true>>>>
  make_atomic_histogram_with_error_adopted(void *rawdata, std::size_t size, Axes&&... axes) {
    using accummulator_type = narf::weighted_sum<double, true>;
    using storage_type = RVecDerived<accummulator_type>;

    static_assert(std::is_standard_layout<accummulator_type>::value);

    // allowed if the relevant classes have standard layout
    accummulator_type *data = reinterpret_cast<accummulator_type*>(rawdata);

    return make_histogram_with(storage_type(data, size), std::forward<Axes>(axes)...);
  }

  template<typename DFType, typename Helper, typename... ColTypes>
  ROOT::RDF::RResultPtr<typename std::decay_t<Helper>::Result_t>
  book_helper(DFType &df, Helper &&helper, const std::vector<std::string> &colnames) {
    return df.template Book<ColTypes...>(std::forward<Helper>(helper), colnames);
  }

}


#endif
