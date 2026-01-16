#ifndef NARF_HISTUTILS_H
#define NARF_HISTUTILS_H

#include <boost/histogram.hpp>
#include "traits.hpp"
#include "atomic_adaptor.hpp"
#include "tensorutils.hpp"
#include <ROOT/RResultPtr.hxx>
#include <ROOT/TThreadExecutor.hxx>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "oneapi/tbb.h"

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

  template <typename... Args>
  boost::histogram::axis::variable<Args...> make_variable_axis(const std::vector<double> &edges) {
    return boost::histogram::axis::variable<Args...>(edges);
  }

  template <typename... Args>
  boost::histogram::axis::category<Args...> make_category_axis(const std::vector<std::string> &cats) {
    return boost::histogram::axis::category<Args...>(cats);
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

  template <class A>
  std::size_t unlinearize_index_impl(const std::size_t out, const std::size_t stride, const A& ax,
                              axis::index_type &idx) noexcept {
    const auto opt = axis::traits::get_options<A>();
    const axis::index_type begin = opt & axis::option::underflow ? -1 : 0;
    const axis::index_type end = opt & axis::option::overflow ? ax.size() + 1 : ax.size();
    const axis::index_type extent = end - begin;
    //TODO refactor this to avoid two separate division/modulus operations here
    idx = begin + (out/stride)%extent;
    return extent;
  }

  template<typename HIST>
  typename HIST::multi_index_type unlinearize_index(const HIST &hist, std::size_t idx) {
    auto indices = HIST::multi_index_type::create(hist.rank());
    using std::begin;
    auto i = begin(indices);
    auto stride = static_cast<std::size_t>(1);
    hist.for_each_axis([&](const auto& a) { stride *= unlinearize_index_impl(idx, stride, a, *i++); });
    return indices;
  }

  template<typename HIST>
  class indexed_range_linear {
  public:
    using indices_t = typename HIST::multi_index_type;
    using value_iterator_t = std::conditional_t<std::is_const<HIST>::value,
                                            typename HIST::const_iterator,
                                            typename HIST::iterator>;

    class iterator {
    public:

      using iterator_category = std::forward_iterator_tag;
      using value_type = typename value_iterator_t::value_type;
      using difference_type = typename value_iterator_t::difference_type;
      using pointer = typename value_iterator_t::pointer;
      using reference = typename value_iterator_t::reference;

      iterator(HIST &hist, std::size_t idx) : indices_(unlinearize_index(hist, idx)),
        indices_begin_(HIST::multi_index_type::create(hist.rank())),
        indices_end_(HIST::multi_index_type::create(hist.rank())) {

        auto itb = indices_begin_.begin();
        auto ite = indices_end_.begin();
        hist.for_each_axis([&itb, &ite](const auto &ax) {
          const auto opt = axis::traits::get_options<std::remove_reference_t<decltype(ax)>>();
          *(itb++) = opt & axis::option::underflow ? -1 : 0;
          *(ite++) = opt & axis::option::overflow ? ax.size() + 1 : ax.size();
        });

        if (idx == hist.size()) {
          iter_ = hist.end();
        }
        else {
          iter_ = hist.begin() + idx;
          // explicitly check that the indices match
          if (&hist.at(indices_) != &*iter_) {
            throw std::runtime_error("inconsistency in construction of indexed_range_linear iterator.");
          }
        }

      }

      reference operator*() const { return *iter_; }

      iterator& operator++() {

        auto it = indices_.begin();
        auto itb = indices_begin_.begin();
        auto ite = indices_end_.begin();

        ++(*it);
        while (it != indices_.end() && *it == *(ite++)) {
          *(it++) = *(itb++);
          if (it == indices_.end()) {
            //this means we're moving to one-past-the-end
            break;
          }
          ++*(it);
        }

        ++iter_;

        return *this;
      }

      bool operator==(const iterator& x) const noexcept { return iter_ == x.iter_; }
      bool operator!=(const iterator& x) const noexcept { return !operator==(x); }

      // make iterator ready for C++17 sentinels
      bool operator==(const value_iterator_t& x) const noexcept { return iter_ == x; }
      bool operator!=(const value_iterator_t& x) const noexcept { return !operator==(x); }

      const indices_t &indices() const noexcept { return indices_; }
      const value_iterator_t &iter() const noexcept { return iter_; }

    private:
      indices_t indices_;
      indices_t indices_begin_;
      indices_t indices_end_;
      value_iterator_t iter_;
    };

    indexed_range_linear(HIST &hist, const std::size_t ibegin, const std::size_t iend) :
      begin_(hist, ibegin), end_(hist, iend) {}

    iterator begin() noexcept { return begin_; }
    iterator end() noexcept { return end_; }

  private:
    iterator begin_;
    iterator end_;
  };

  template<typename HIST>
  auto indexed_linear(HIST &&hist) {
    using range_t = indexed_range_linear<std::remove_reference_t<HIST>>;
    return range_t(hist, 0, hist.size());
  }

  template<typename HIST>
  auto indexed_linear(HIST &&hist, const std::size_t ibegin, const std::size_t iend) {
    using range_t = indexed_range_linear<std::remove_reference_t<HIST>>;
    return range_t(hist, ibegin, iend);
  }

  template<typename T, std::size_t NDims, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
  class array_interface_view {

    // TODO memcpy cast directly to accumulator types instead of underlying values to avoid hardcoded offsets?

  public:
    
    static constexpr bool is_weighted_sum() { return acc_traits<T>::is_weighted_sum; }

    // TODO handle other index types for constructor?
    array_interface_view(void *buffer, const std::vector<std::ptrdiff_t> &sizes, const std::vector<std::ptrdiff_t> &strides, const std::vector<bool> &underflow) :
      data_(static_cast<std::byte*>(buffer)) {

        std::copy(sizes.begin(), sizes.end(), sizes_.begin());
        std::copy(strides.begin(), strides.end(), strides_.begin());
        std::copy(underflow.begin(), underflow.end(), flow_offsets_.begin());
    }

    std::ptrdiff_t size() const { return std::accumulate(sizes_.begin(), sizes_.end(), 1, std::multiplies<std::ptrdiff_t>()); }

    template <typename HIST>
    void from_boost(const HIST &hist) {

      // to avoid explicitly checking if bins are in range, require that all axes
      // in the view are at least as large as for the C++ histogram
      // note this doesn't need to be checked for the tensor axes
      // since they have no underflow or overflow in the C++ histogram and are safe by
      //construction.

      // TODO make this check more explicit in terms of overflow/underflow


      for (std::size_t iax = 0; iax < hist.rank(); ++iax) {
        if (boost::histogram::axis::traits::extent(hist.axis(iax)) > sizes_[iax]) {
            throw std::runtime_error("Incompatible underflow/overflow for histogram conversion");
        }
      }

      constexpr std::size_t rank = NDims;

      using acc_t = typename HIST::storage_type::value_type;
      using acc_trait = narf::acc_traits<acc_t>;

      auto fill_hist = [&hist](auto &fill_bin) {
        if (ROOT::IsImplicitMTEnabled()) {
          auto rarena = ROOT::Internal::GetGlobalTaskArena(ROOT::GetThreadPoolSize());
          //FIXME this is ugly but this really does inherit from tbb::task_arena even though it's hidden from the compiler
          tbb::task_arena &arena = *reinterpret_cast<tbb::task_arena*>(&rarena->Access());
          using brange_t = oneapi::tbb::blocked_range<std::size_t>;
          brange_t brange(0, hist.size());
          auto pfill = [&fill_bin, &hist](const brange_t& r) {
            auto range = indexed_linear(hist, r.begin(), r.end());
            for (auto it = range.begin(); it != range.end(); ++it) {
              fill_bin(it);
            }
          };
          arena.execute([&] {
              oneapi::tbb::this_task_arena::isolate([&] {
                oneapi::tbb::parallel_for(brange, pfill);
              });
          });
        }
        else {
          auto range = indexed_linear(hist);
          for (auto it = range.begin(); it != range.end(); ++it) {
            fill_bin(it);
          }
        }
      };

      if constexpr (acc_trait::is_tensor) {
        const auto fillrank = hist.rank();

        auto fill_bin = [this, &fillrank](auto const &it) {

          std::array<std::ptrdiff_t, rank> idxs;
          auto itidx = it.indices().begin();
          for (std::size_t idim = 0; idim < fillrank; ++idim) {
            idxs[idim] = *(itidx++) + flow_offsets_[idim];
          }

          auto const &tensor_acc_val = *it;

          for (auto it = tensor_acc_val.indices_begin(); it != tensor_acc_val.indices_end(); ++it) {
            const auto tensor_indices = it.indices;
            for (std::size_t idim = fillrank; idim < rank; ++idim) {
              idxs[idim] = tensor_indices[idim - fillrank] + flow_offsets_[idim];
            }

            auto const &acc_val = std::apply(tensor_acc_val.data(), tensor_indices);

            std::byte *elem = element_ptr(idxs);

            T acc_val_tmp = acc_val;
            std::memcpy(elem, &acc_val_tmp, sizeof(T));
          }
        };

        fill_hist(fill_bin);

      }
      else {
        auto fill_bin = [this](auto const &it) {
          std::array<std::ptrdiff_t, rank> idxs;
          auto itidx = it.indices().begin();
          for (std::size_t idim = 0; idim < rank; ++idim) {
            idxs[idim] = *(itidx++) + flow_offsets_[idim];
          }

          auto const &acc_val = *it;

          std::byte *elem = element_ptr(idxs);

          const T acc_val_tmp = acc_val;
          std::memcpy(elem, &acc_val_tmp, sizeof(T));
        };

        fill_hist(fill_bin);
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

        for (auto&& x: indexed(hist, coverage::all)) {
          std::array<std::ptrdiff_t, rank> idxs;
          for (std::size_t idim = 0; idim < fillrank; ++idim) {
            idxs[idim] = x.index(idim) + flow_offsets_[idim];
          }

          auto &tensor_acc_val = *x;

          for (auto it = tensor_acc_val.indices_begin(); it != tensor_acc_val.indices_end(); ++it) {
            const auto tensor_indices = it.indices;
            for (std::size_t idim = fillrank; idim < rank; ++idim) {
              idxs[idim] = tensor_indices[idim - fillrank] + flow_offsets_[idim];
            }

            // overflow or underflow bin in the histogram with no corresponding bin in the view
            if (!in_range(idxs)) {
              continue;
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
        for (auto&& x: indexed(hist, coverage::all)) {
          std::array<std::ptrdiff_t, rank> idxs;
          for (std::size_t idim = 0; idim < rank; ++idim) {
            idxs[idim] = x.index(idim) + flow_offsets_[idim];
          }

          // overflow or underflow bin in the histogram with no corresponding bin in the view
          if (!in_range(idxs)) {
            continue;
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
          actual_idxs[idim] = idxs[idim] - 1 + flow_offsets_[idim];
        }
        
        // overflow or underflow bin in the histogram with no corresponding bin in the view
        if (!in_range(actual_idxs)) {
          continue;
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
          actual_idxs[idim] = idxs[idim] - 1 + flow_offsets_[idim];
        }
        
        // overflow or underflow bin in the histogram with no corresponding bin in the view
        if (!in_range(actual_idxs)) {
          continue;
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
    
    bool in_range(const std::array<std::ptrdiff_t, NDims> &idxs) const {
      for (std::size_t idim = 0; idim < NDims; ++idim) {
        if (idxs[idim] < 0 || idxs[idim] >= sizes_[idim]) {
          return false;
        }
      }

      return true;
    }

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
    std::array<std::ptrdiff_t, NDims> flow_offsets_;

  };

  template<typename T, std::size_t NDims>
  struct is_array_interface_view<array_interface_view<T, NDims>> : public std::true_type{
  };


  // helper for bin lookup which implements the compile-time loop over axes
  template<typename HIST, typename... Xs, std::size_t... Idxs>
  auto const &get_value_impl(const HIST &hist, std::index_sequence<Idxs...>, const Xs&... xs) {
      return hist.at(hist.template axis<Idxs>().index(xs)...);
  }

  // variadic templated bin lookup
  template<typename HIST, typename... Xs>
  auto const &get_value(const HIST &hist, const Xs&... xs) {
      return get_value_impl(hist, std::index_sequence_for<Xs...>{}, xs...);
  }

  // Helper which holds a histogram and facilitates bin content lookup
  // RDataFrame still needs explicit non-templated operator() arguments for now
  template <typename Storage, typename... Axes>
  class HistHelper {
  protected:
    using hist_t = boost::histogram::histogram<std::tuple<Axes...>, Storage>;

  public:
    HistHelper(hist_t &&resource) : resourceHist_(std::make_shared<const hist_t>(std::move(resource))) {}

    double operator()(const boost::histogram::axis::traits::value_type<Axes>&... args) {
      return narf::get_value(*resourceHist_, args...);
    }

  protected:
    std::shared_ptr<const hist_t> resourceHist_;
  };

  // CTAD doesn't work reliably from cppyy so add factory function
  template <typename Storage, typename... Axes>
  HistHelper<Storage, Axes...> make_hist_helper(boost::histogram::histogram<std::tuple<Axes...>, Storage> &&h) {
    using hist_t = boost::histogram::histogram<std::tuple<Axes...>, Storage>;
    return HistHelper(std::forward<hist_t>(h));
  }

  // Helper which facilitates conversion from value to quantile for a single variable
  // The underlying histogram holds a tensor with the bin edges for the quantiles in the last variable,
  // conditional on all the previous variables
  template <typename Storage, typename... Axes>
  class QuantileHelper : public HistHelper<Storage, Axes...> {
    using base_t = HistHelper<Storage, Axes...>;
    using hist_t = typename base_t::hist_t;
    using scalar_t = typename Storage::value_type::tensor_t::Scalar;
    static constexpr auto nquants = Storage::value_type::size;

  public:
    QuantileHelper(hist_t &&resource) : base_t(std::forward<hist_t>(resource)) {}

    boost::histogram::axis::index_type operator()(const boost::histogram::axis::traits::value_type<Axes>&... args, const scalar_t &last) {
      auto const &hist = *base_t::resourceHist_;
      auto const &edges = narf::get_value(hist, args...).data();

      // find the quantile bin corresponding to the last argument
      auto const upper = std::upper_bound(edges.data(), edges.data()+nquants, last);
      auto const iquant = std::distance(edges.data(), upper);
      // auto const lower = std::lower_bound(edges.data(), edges.data()+nquants, last);
      // auto const iquant = std::distance(edges.data(), lower) - 1;
      return std::clamp<boost::histogram::axis::index_type>(iquant, 0, nquants-1);
    }
  };

  // CTAD doesn't work reliably from cppyy so add factory function
  // also use std::vector as input to play nice with cppyy auto conversion from python types
  // template <typename Storage, typename... Axes>
  // HistHelper<Storage, Axes...> make_quantile_helper(boost::histogram::histogram<std::tuple<Axes...>, Storage> &&h, const std::array<double, Storage::value_type::size> &quants) {
  //   using hist_t = boost::histogram::histogram<std::tuple<Axes...>, Storage>;
  //   return QuantileHelper(std::forward<hist_t>(h), quants);
  // }

  template <typename Storage, typename... Axes>
  QuantileHelper<Storage, Axes...> make_quantile_helper(boost::histogram::histogram<std::tuple<Axes...>, Storage> &&h) {
    using hist_t = boost::histogram::histogram<std::tuple<Axes...>, Storage>;
    return QuantileHelper<Storage, Axes...>(std::forward<hist_t>(h));
  }

#if false

  // base template
  template <typename... Hists>
  class MultiQuantileHelper {};

  // specialization (to allow multiple parameter packs)
  template<typename... Hists, typename StorageLast, typename... AxesLast>
  class MultiQuantileHelper<boost::histogram::histogram<std::tuple<AxesLast...>, StorageLast>, Hists...> {
    using scalar_t = typename StorageLast::value_type::tensor_t::Scalar;
    using last_hist_t = boost::histogram::histogram<std::tuple<AxesLast...>, StorageLast>;
    using index_t = boost::histogram::axis::index_type;
    static auto constexpr nhelpers = sizeof...(Hists) + 1;
    static auto constexpr nargs = sizeof...(AxesLast) + 1;
    static auto constexpr nargscond = nargs - nhelpers;

  public:
    MultiQuantileHelper(Hists&&... hists, last_hist_t&& histlast) : helpers_(QuantileHelper(std::move(hists))..., QuantileHelper<StorageLast, AxesLast...>(std::move(histlast))) {}

    std::array<index_t, nhelpers> operator()(const boost::histogram::axis::traits::value_type<AxesLast>&... args, const scalar_t &last) {
      auto const argtup = std::tie(args..., last);
      return getquants(argtup, std::array<index_t, 0>{}, std::make_index_sequence<nargscond>{}, std::make_index_sequence<0>{});
    }

  private:
    template<std::size_t... IdxsArgsCond, std::size_t... IdxsQuants, typename Tup>
    auto getquants(const Tup &argtup,  const std::array<index_t, sizeof...(IdxsQuants)> &quants, std::index_sequence<IdxsArgsCond...>, std::index_sequence<IdxsQuants...>) {
      // auto constexpr nidxs = sizeof...(Idxs);
      auto constexpr nin = sizeof...(IdxsQuants);
      auto constexpr nout = nin + 1;
      std::array<index_t, nout> res;
      std::copy(quants.begin(), quants.end(), res.begin());
      res[nin] = std::get<nin>(helpers_)(std::get<IdxsArgsCond>(argtup)..., std::get<IdxsQuants>(quants)..., std::get<nargscond + nin>(argtup));

      if constexpr (nout < nhelpers) {
        return getquants(argtup, res, std::make_index_sequence<nargscond>{}, std::make_index_sequence<nout>{});
      }
      else {
        return res;
      }
    }
    std::tuple<decltype(QuantileHelper(Hists()))..., QuantileHelper<StorageLast, AxesLast...>> helpers_;
  };

  template<typename... Hists, typename StorageLast, typename... AxesLast>
  MultiQuantileHelper<boost::histogram::histogram<std::tuple<AxesLast...>, StorageLast>, Hists...> make_multi_quantile_helper(Hists&&... hists, boost::histogram::histogram<std::tuple<AxesLast...>, StorageLast> &&histlast) {
    using last_hist_t = boost::histogram::histogram<std::tuple<AxesLast...>, StorageLast>;
    return MultiQuantileHelper<last_hist_t, Hists...>(std::forward<Hists>(hists)..., std::forward<last_hist_t>(histlast));
  }

#endif

  // CTAD doesn't work reliably from cppyy so add factory function
  // also use std::vector as input to play nice with cppyy auto conversion from python types
  // template <typename Storage, typename... Axes>
  // QuantileHelper<Storage, Axes...> make_quantile_helper(boost::histogram::histogram<std::tuple<Axes...>, Storage> &&h, const std::vector<double> &quantsv) {
  //   using hist_t = boost::histogram::histogram<std::tuple<Axes...>, Storage>;
  //   constexpr auto nquants = Storage::value_type::size;
  //
  //   std::array<double, nquants> quants;
  //   std::copy(quantsv.begin(), quantsv.end(), quants.begin());
  //   return QuantileHelper(std::forward<hist_t>(h), quants);
  // }

  // template class QuantileHelper<boost::histogram::storage_adaptor<vector<narf::tensor_accumulator<double,Eigen::Sizes<10> > > >, boost::histogram::axis::regular<double,boost::use_default,boost::use_default,boost::histogram::axis::option::bitset<3> >,boost::histogram::axis::regular<double,boost::use_default,boost::use_default,boost::histogram::axis::option::bitset<3> >>;

  void testqhelper() {
    using hist_t = boost::histogram::histogram<tuple<boost::histogram::axis::regular<double,boost::use_default,boost::use_default,boost::histogram::axis::option::bitset<3> >,boost::histogram::axis::regular<double,boost::use_default,boost::use_default,boost::histogram::axis::option::bitset<3> > >,boost::histogram::storage_adaptor<vector<narf::tensor_accumulator<double,Eigen::Sizes<10> > > > >;


    // std::array<double, 10> arr;
    // std::vector<double> arr(10);
    // hist_t h();
    auto helper = make_quantile_helper(hist_t());

    // auto multihelper = make_multi_quantile_helper(hist_t(), hist_t());
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
