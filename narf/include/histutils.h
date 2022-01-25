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

  template<typename T, typename A, typename... Axes>
  histogram<std::tuple<std::decay_t<Axes>...>, adopted_storage<T>>
  make_histogram_adopted(bool do_init, A addr, std::size_t buf_size, Axes&&... axes) {
    void *buffer = reinterpret_cast<void*>(addr);
    adopted_storage<T> storage(do_init, buffer, buf_size);
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

  template <typename HIST>
  void fill_boost(const HIST &hist, double* vals, double *vars, const std::vector<int> &stridevals, const std::vector<int> &stridevars) {
//     double *vals = reinterpret_cast<double*>(addrvals);
//     double *vars = reinterpret_cast<double*>(addrvars);

    const auto rank = stridevals.size();

    std::vector<std::size_t> stridevalsarr(rank);
    std::vector<std::size_t> stridevarsarr(rank);
    for (unsigned int iaxis = 0; iaxis < rank; ++iaxis) {
      stridevalsarr[iaxis] = stridevals[iaxis]/sizeof(double);
      if (vars != nullptr) {
        stridevarsarr[iaxis] = stridevars[iaxis]/sizeof(double);
      }
    }

    // has to be at least 3 for TH1 case
    std::vector<int> idxs(std::max(rank, static_cast<decltype(rank)>(3)));
    const auto nbins = get_n_bins(hist);
    for (std::decay_t<decltype(nbins)> ibin = 0; ibin < nbins; ++ibin) {
      fill_idxs(hist, ibin, idxs);

      std::size_t offsetval = 0;
      std::size_t offsetvar = 0;
      for (unsigned int iaxis = 0; iaxis < rank; ++iaxis) {
        offsetval += stridevalsarr[iaxis]*idxs[iaxis];
        if (vars != nullptr) {
          offsetvar += stridevarsarr[iaxis]*idxs[iaxis];
        }
      }

      vals[offsetval] = hist.GetBinContent(ibin);
      if (vars != nullptr) {
        vars[offsetvar] = get_bin_error2(hist, ibin);
      }
    }
  }

  template <typename HIST>
  void fill_root(HIST &hist, double* vals, double *vars, const std::vector<int> &stridevals, const std::vector<int> &stridevars) {
//     double *vals = reinterpret_cast<double*>(addrvals);
//     double *vars = reinterpret_cast<double*>(addrvars);

    const auto rank = stridevals.size();

    std::vector<std::size_t> stridevalsarr(rank);
    std::vector<std::size_t> stridevarsarr(rank);
    for (unsigned int iaxis = 0; iaxis < rank; ++iaxis) {
      stridevalsarr[iaxis] = stridevals[iaxis]/sizeof(double);
      if (vars != nullptr) {
        stridevarsarr[iaxis] = stridevars[iaxis]/sizeof(double);
      }
    }

    // has to be at least 3 for TH1 case
    std::vector<int> idxs(std::max(rank, static_cast<decltype(rank)>(3)));
    const auto nbins = get_n_bins(hist);
    for (std::decay_t<decltype(nbins)> ibin = 0; ibin < nbins; ++ibin) {
      fill_idxs(hist, ibin, idxs);

      std::size_t offsetval = 0;
      std::size_t offsetvar = 0;
      for (unsigned int iaxis = 0; iaxis < rank; ++iaxis) {
        offsetval += stridevalsarr[iaxis]*idxs[iaxis];
        if (vars != nullptr) {
          offsetvar += stridevarsarr[iaxis]*idxs[iaxis];
        }
      }

      hist.SetBinContent(ibin, vals[offsetval]);
      if (vars != nullptr) {
        set_bin_error2(hist, ibin, vars[offsetvar]);
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
