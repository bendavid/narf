#ifndef NARF_HISTUTILS_H
#define NARF_HISTUTILS_H

#include <boost/histogram.hpp>
#include "traits.h"
#include "atomic_adaptor.h"
#include "tensorutils.h"
#include <ROOT/RResultPtr.hxx>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace narf {
  using namespace boost::histogram;

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

  template <typename A>
  void fill_idxs(const TH1& hist, int ibin, A &idxs) {
    hist.GetBinXYZ(ibin, idxs[0], idxs[1], idxs[2]);
  }

  template <typename A>
  void fill_idxs(const THnBase& hist, Long64_t ibin, A &idxs) {
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

  template<typename T, std::size_t NDims, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
  class array_interface_view {

    // TODO memcpy cast directly to accumulator types instead of underlying values to avoid hardcoded offsets?

  public:

    // TODO handle other index types for constructor?
    array_interface_view(void *buffer, const std::vector<int> &sizes, const std::vector<int> &strides) :
      data_(static_cast<std::byte*>(buffer)) {

        for (std::size_t idim = 0; idim < NDims; ++idim) {
          sizes_[idim] = sizes[idim];
          strides_[idim] = strides[idim];
        }

    }

    std::ptrdiff_t size() const { return std::accumulate(sizes_.begin(), sizes_.end(), 1, std::multiplies<std::ptrdiff_t>()); }

    template <typename HIST>
    void from_boost(HIST &hist) {
      //TODO multithreading for this

      constexpr std::size_t rank = NDims;

      using acc_t = typename HIST::storage_type::value_type;
      using acc_trait = narf::acc_traits<acc_t>;



      if constexpr (acc_trait::is_tensor) {
        const auto fillrank = hist.rank();

        std::vector<ptrdiff_t> flow_offsets(fillrank);
        for (std::size_t idim = 0; idim < fillrank; ++idim) {
          flow_offsets[idim] = boost::histogram::axis::traits::options(hist.axis(idim)) & boost::histogram::axis::option::underflow ? 1 : 0;
        }

        for (auto&& x: indexed(hist, coverage::all)) {
          std::array<std::ptrdiff_t, rank> idxs;
          for (std::size_t idim = 0; idim < fillrank; ++idim) {
            idxs[idim] = x.index(idim) + flow_offsets[idim];
          }

          auto const &tensor_acc_val = *x;

          for (auto it = tensor_acc_val.indices_begin(); it != tensor_acc_val.indices_end(); ++it) {
            const auto tensor_indices = it.indices;
            for (std::size_t idim = fillrank; idim < rank; ++idim) {
              idxs[idim] = tensor_indices[idim - fillrank];
            }

            auto const &acc_val = std::apply(tensor_acc_val.data(), tensor_indices);

            std::byte *elem = element_ptr(idxs);

            T acc_val_tmp = acc_val;
            std::memcpy(elem, &acc_val_tmp, sizeof(T));
          }
        }
      }
      else {
        std::array<ptrdiff_t, rank> flow_offsets;
        for (std::size_t idim = 0; idim < rank; ++idim) {
          flow_offsets[idim] = boost::histogram::axis::traits::options(hist.axis(idim)) & boost::histogram::axis::option::underflow ? 1 : 0;
        }

        for (auto&& x: indexed(hist, coverage::all)) {
          std::array<std::ptrdiff_t, rank> idxs;
          for (std::size_t idim = 0; idim < rank; ++idim) {
            idxs[idim] = x.index(idim) + flow_offsets[idim];
          }

          auto const &acc_val = *x;

          std::byte *elem = element_ptr(idxs);

          const T acc_val_tmp = acc_val;
          std::memcpy(elem, &acc_val_tmp, sizeof(T));
        }
      }
    }

    template <typename HIST>
    void to_boost(HIST &hist) const {

      //TODO multithreading for this

      constexpr std::size_t rank = NDims;

      using acc_t = typename HIST::storage_type::value_type;
      using acc_trait = narf::acc_traits<acc_t>;



      if constexpr (acc_trait::is_tensor) {
        const auto fillrank = hist.rank();

        std::vector<ptrdiff_t> flow_offsets(fillrank);
        for (std::size_t idim = 0; idim < fillrank; ++idim) {
          flow_offsets[idim] = boost::histogram::axis::traits::options(hist.axis(idim)) & boost::histogram::axis::option::underflow ? 1 : 0;
        }

        for (auto&& x: indexed(hist, coverage::all)) {
          std::array<std::ptrdiff_t, rank> idxs;
          for (std::size_t idim = 0; idim < fillrank; ++idim) {
            idxs[idim] = x.index(idim) + flow_offsets[idim];
          }

          auto &tensor_acc_val = *x;

          for (auto it = tensor_acc_val.indices_begin(); it != tensor_acc_val.indices_end(); ++it) {
            const auto tensor_indices = it.indices;
            for (std::size_t idim = fillrank; idim < rank; ++idim) {
              idxs[idim] = tensor_indices[idim - fillrank];
            }

            auto &acc_val = std::apply(tensor_acc_val.data(), tensor_indices);

            const std::byte *elem = element_ptr(idxs);

            T acc_val_tmp;
            std::memcpy(&acc_val_tmp, elem, sizeof(T));
            acc_val = acc_val_tmp;
          }
        }
      }
      else {
        std::array<ptrdiff_t, rank> flow_offsets;
        for (std::size_t idim = 0; idim < rank; ++idim) {
          flow_offsets[idim] = boost::histogram::axis::traits::options(hist.axis(idim)) & boost::histogram::axis::option::underflow ? 1 : 0;
        }

        for (auto&& x: indexed(hist, coverage::all)) {
          std::array<std::ptrdiff_t, rank> idxs;
          for (std::size_t idim = 0; idim < rank; ++idim) {
            idxs[idim] = x.index(idim) + flow_offsets[idim];
          }

          auto &acc_val = *x;

          const std::byte *elem = element_ptr(idxs);

          T acc_val_tmp;
          std::memcpy(&acc_val_tmp, elem, sizeof(T));
          acc_val = acc_val_tmp;
        }
      }
    }

    template <typename HIST>
    void from_root(HIST &hist) {
      constexpr std::size_t rank = NDims;

      // has to be at least 3 for TH1 case
      constexpr std::size_t arr_size = std::max(rank, static_cast<decltype(rank)>(3));

      const auto nbins = get_n_bins(hist);
      for (std::decay_t<decltype(nbins)> ibin = 0; ibin < nbins; ++ibin) {

        std::array<int, arr_size> idxs{};
        fill_idxs(hist, ibin, idxs);

        // different type and might be different size
        std::array<std::ptrdiff_t, NDims> actual_idxs;
        for (std::size_t idim = 0; idim < rank; ++idim) {
          actual_idxs[idim] = idxs[idim];
        }

        std::byte *elem = element_ptr(actual_idxs);

        if constexpr (acc_traits<T>::is_weighted_sum) {
          const T acc_val_tmp(hist.GetBinContent(ibin), get_bin_error2(hist, ibin));
          std::memcpy(elem, &acc_val_tmp, sizeof(T));
        }
        else {
          const T acc_val_tmp = hist.GetBinContent(ibin);
          std::memcpy(elem, &acc_val_tmp, sizeof(T));
        }
      }
    }


    template <typename HIST>
    void to_root(HIST &hist) const {
      constexpr std::size_t rank = NDims;

      // has to be at least 3 for TH1 case
      constexpr std::size_t arr_size = std::max(rank, static_cast<decltype(rank)>(3));

      const auto nbins = get_n_bins(hist);
      for (std::decay_t<decltype(nbins)> ibin = 0; ibin < nbins; ++ibin) {

        std::array<int, arr_size> idxs{};
        fill_idxs(hist, ibin, idxs);

        // different type and might be different size
        std::array<std::ptrdiff_t, NDims> actual_idxs;
        for (std::size_t idim = 0; idim < rank; ++idim) {
          actual_idxs[idim] = idxs[idim];
        }

        const std::byte *elem = element_ptr(actual_idxs);

        T acc_val_tmp;
        std::memcpy(&acc_val_tmp, elem, sizeof(T));

        if constexpr (acc_traits<T>::is_weighted_sum) {
          hist.SetBinContent(ibin, acc_val_tmp.value());
          set_bin_error2(hist, ibin, acc_val_tmp.variance());

        }
        else {
          hist.SetBinContent(ibin, acc_val_tmp);
        }
      }
    }


  private:

    std::ptrdiff_t element_offset(const std::array<std::ptrdiff_t, NDims> &idxs) const {
      std::ptrdiff_t offset = 0;
      for (std::size_t idim = 0; idim < NDims; ++idim) {
        offset += idxs[idim]*strides_[idim];
      }
      return offset;
    }

    std::byte *element_ptr(const std::array<std::ptrdiff_t, NDims> &idxs) {
      return data_ + element_offset(idxs);
    }

    const std::byte *element_ptr(const std::array<std::ptrdiff_t, NDims> &idxs) const {
      return data_ + element_offset(idxs);
    }

    std::byte *data_;
    std::array<std::ptrdiff_t, NDims> sizes_;
    std::array<std::ptrdiff_t, NDims> strides_;

  };

  template<typename T, std::size_t NDims>
  struct is_array_interface_view<array_interface_view<T, NDims>> : public std::true_type{
  };

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
