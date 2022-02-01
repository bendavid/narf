#ifndef NARF_HISTUTILS_H
#define NARF_HISTUTILS_H

#include <boost/histogram.hpp>
#include "adopted_storage.h"
#include "traits.h"
#include "atomic_adaptor.h"
#include "tensorutils.h"
#include <ROOT/RResultPtr.hxx>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

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
  histogram<std::tuple<std::decay_t<Axis>, std::decay_t<Axes>...>, dense_storage<narf::atomic_adaptor<double>>>
  make_atomic_histogram(Axis&& axis, Axes&&... axes) {
    return make_histogram_with(dense_storage<narf::atomic_adaptor<double>>(), std::forward<Axis>(axis), std::forward<Axes>(axes)...);
  }

  template<typename Axis, typename... Axes>
  histogram<std::tuple<std::decay_t<Axis>, std::decay_t<Axes>...>, dense_storage<boost::histogram::accumulators::weighted_sum<double>>>
  make_histogram_with_error(Axis&& axis, Axes&&... axes) {
    return make_histogram_with(dense_storage<boost::histogram::accumulators::weighted_sum<double>>(), std::forward<Axis>(axis), std::forward<Axes>(axes)...);
  }

  template<typename Axis, typename... Axes>
  histogram<std::tuple<std::decay_t<Axis>, std::decay_t<Axes>...>, dense_storage<narf::atomic_adaptor<boost::histogram::accumulators::weighted_sum<double>>>>
  make_atomic_histogram_with_error(Axis&& axis, Axes&&... axes) {
    return make_histogram_with(dense_storage<narf::atomic_adaptor<boost::histogram::accumulators::weighted_sum<double>>>(), std::forward<Axis>(axis), std::forward<Axes>(axes)...);
  }

  template<typename Storage, typename... Axes>
  histogram<std::tuple<std::decay_t<Axes>...>, Storage>
  make_histogram_with_storage(Storage &&storage, Axes&&... axes) {
    return make_histogram_with(std::forward<Storage>(storage), std::forward<Axes>(axes)...);
  }

  template<typename Storage, typename... Axes>
  histogram<std::tuple<std::decay_t<Axes>...>, storage_adaptor<Storage>>
  make_histogram_with_adaptable(Storage &&storage, Axes&&... axes) {
    return make_histogram_with(std::forward<Storage>(storage), std::forward<Axes>(axes)...);
  }

  template<typename T, typename... Axes>
  histogram<std::tuple<std::decay_t<Axes>...>, dense_storage<T>>
  make_histogram_dense(Axes&&... axes) {
    return make_histogram_with(dense_storage<T>(), std::forward<Axes>(axes)...);
  }

  template<typename T, bool do_init, typename... Axes>
  histogram<std::tuple<std::decay_t<Axes>...>, adopted_storage<T, do_init>>
  make_histogram_adopted(void *buffer, std::size_t buf_size, Axes&&... axes) {
    adopted_storage<T, do_init> storage(buffer, buf_size);
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

  double get_bin_error2(const TH1& hist, int ibin) {
    const double err = hist.GetBinError(ibin);
    return err*err;
  }

  double get_bin_error2(const THnBase& hist, Long64_t ibin) {
    return hist.GetBinError2(ibin);
  }

  void fill_idxs(const TH1& hist, int ibin, std::vector<int> &idxs) {
    hist.GetBinXYZ(ibin, idxs[0], idxs[1], idxs[2]);
  }

  void fill_idxs(const THnBase& hist, Long64_t ibin, std::vector<int> &idxs) {
    hist.GetBinContent(ibin, idxs.data());
  }

  int get_n_bins(const TH1& hist) {
    return hist.GetNcells();
  }

  Long64_t get_n_bins(const THnBase& hist) {
    return hist.GetNbins();
  }

  void set_bin_error2(TH1& hist, int ibin, double var) {
    hist.SetBinError(ibin, std::sqrt(var));
  }

  void set_bin_error2(THnBase& hist, Long64_t ibin, double var) {
    hist.SetBinError2(ibin, var);
  }

  template <typename HIST, typename val_t = double, typename var_t = val_t>
  void fill_boost(const HIST &hist, void* vals, void *vars, const std::vector<int> &stridevals, const std::vector<int> &stridevars) {
    std::byte *valbytes = static_cast<std::byte*>(vals);
    std::byte *varbytes = static_cast<std::byte*>(vars);

    const auto rank = stridevals.size();

    // has to be at least 3 for TH1 case
    std::vector<int> idxs(std::max(rank, static_cast<decltype(rank)>(3)));
    const auto nbins = get_n_bins(hist);
    for (std::decay_t<decltype(nbins)> ibin = 0; ibin < nbins; ++ibin) {
      fill_idxs(hist, ibin, idxs);

      std::size_t offsetval = 0;
      std::size_t offsetvar = 0;
      for (unsigned int iaxis = 0; iaxis < rank; ++iaxis) {
        offsetval += stridevals[iaxis]*idxs[iaxis];
        if (vars != nullptr) {
          offsetvar += stridevars[iaxis]*idxs[iaxis];
        }
      }

      *view<val_t>(valbytes + offsetval, sizeof(val_t)) = hist.GetBinContent(ibin);
      if (vars != nullptr) {
        *view<var_t>(varbytes + offsetvar, sizeof(var_t)) = get_bin_error2(hist, ibin);
      }
    }
  }

  template <typename HIST, typename val_t = double, typename var_t = val_t>
  void fill_root(HIST &hist, const void* vals, const void *vars, const std::vector<int> &stridevals, const std::vector<int> &stridevars) {
    const std::byte *valbytes = static_cast<const std::byte*>(vals);
    const std::byte *varbytes = static_cast<const std::byte*>(vars);

    const auto rank = stridevals.size();

    // has to be at least 3 for TH1 case
    std::vector<int> idxs(std::max(rank, static_cast<decltype(rank)>(3)));
    const auto nbins = get_n_bins(hist);
    for (std::decay_t<decltype(nbins)> ibin = 0; ibin < nbins; ++ibin) {
      fill_idxs(hist, ibin, idxs);

      std::size_t offsetval = 0;
      std::size_t offsetvar = 0;
      for (unsigned int iaxis = 0; iaxis < rank; ++iaxis) {
        offsetval += stridevals[iaxis]*idxs[iaxis];
        if (vars != nullptr) {
          offsetvar += stridevars[iaxis]*idxs[iaxis];
        }
      }

      hist.SetBinContent(ibin, bit_cast_ptr<val_t>(valbytes + offsetval));
      if (vars != nullptr) {
        set_bin_error2(hist, ibin, bit_cast_ptr<var_t>(varbytes + offsetval));
      }
    }
  }

  template <typename HIST>
  bool check_storage_order(const HIST &hist, const std::vector<int> &strides) {
    const std::vector<int> origin(strides.size(), 0);
    const auto *origin_addr = &hist.at(origin);
    for (std::size_t idim; idim <strides.size(); ++idim) {
      std::vector<int> coords(strides.size(), 0);
      coords[idim] = 1;
      const auto *addr = &hist.at(coords);
      const ptrdiff_t addr_diff = addr - origin_addr;
      if (addr_diff != strides[idim]) {
        return false;
      }
    }

    return true;
  }

}

// template <typename T, typename... Args>
// Eigen::TensorFixedSize<narf::atomic_adaptor<T>, Args...> &operator+=(Eigen::TensorFixedSize<narf::atomic_adaptor<T>, Args...> &lhs, const Eigen::TensorFixedSize<T, Args...> &rhs) {
//   lhs += rhs.template cast<narf::atomic_adaptor<T>>();
//   return lhs;
// }

// template <typename T, typename... Args>
// operator Eigen::TensorFixedSize<narf::atomic_adaptor<T>, Args...> (const Eigen::TensorFixedSize<narf::atomic_adaptor<T>, Args...> &rhs) {
//   return rhs.template cast<narf::atomic_adaptor<T>>();
// }



#endif
