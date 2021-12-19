#ifndef FILLBOOSTHELPERATOMIC_H
#define FILLBOOSTHELPERATOMIC_H

#include <boost/histogram.hpp>

#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RDF/ActionHelpers.hxx"

#include <iostream>
#include <array>

namespace narf {

   using namespace ROOT::Internal::RDF;
   using namespace boost::histogram;

   template <typename HIST>
   class FillBoostHelperAtomic : public ROOT::Detail::RDF::RActionImpl<FillBoostHelperAtomic<HIST>> {
      std::shared_ptr<HIST> fObject;

      // class which wraps a pointer and implements a no-op increment operator
      template <typename T>
      class ScalarConstIterator {
         const T *obj_;

      public:
         ScalarConstIterator(const T *obj) : obj_(obj) {}
         const T &operator*() const { return *obj_; }
         ScalarConstIterator<T> &operator++() { return *this; }
      };

      // helper functions which provide one implementation for scalar types and another for containers
      // TODO these could probably all be replaced by inlined lambdas and/or constexpr if statements
      // in c++17 or later

      // return unchanged value for scalar
      template <typename T, typename std::enable_if<!IsDataContainer<T>::value, int>::type = 0>
      ScalarConstIterator<T> MakeBegin(const T &val)
      {
         return ScalarConstIterator<T>(&val);
      }

      // return iterator to beginning of container
      template <typename T, typename std::enable_if<IsDataContainer<T>::value, int>::type = 0>
      auto MakeBegin(const T &val)
      {
         return std::begin(val);
      }

      // return 1 for scalars
      template <typename T, typename std::enable_if<!IsDataContainer<T>::value, int>::type = 0>
      std::size_t GetSize(const T &)
      {
         return 1;
      }

      // return container size
      template <typename T, typename std::enable_if<IsDataContainer<T>::value, int>::type = 0>
      std::size_t GetSize(const T &val)
      {
   #if __cplusplus >= 201703L
         return std::size(val);
   #else
         return val.size();
   #endif
      }

      template <typename A, typename T, T... Idxs>
      void FillHist(const A &tup, std::integer_sequence<T, Idxs...>) {
         using namespace boost::histogram;
         auto &thisSlotH = *fObject;

         constexpr unsigned int N = std::tuple_size<A>::value;
         thisSlotH(std::get<Idxs>(tup)..., weight(std::get<N-1>(tup)));
      }

      template <typename A, typename T, T... Idxs>
      void FillHistIt(const A &tup, std::integer_sequence<T, Idxs...>) {
         using namespace boost::histogram;
         auto &thisSlotH = *fObject;

         constexpr unsigned int N = std::tuple_size<A>::value;

         thisSlotH(*std::get<Idxs>(tup)..., weight(*std::get<N-1>(tup)));
      }

      template <std::size_t ColIdx, typename End_t, typename... Its>
      void ExecLoop(unsigned int slot, End_t end, Its... its)
      {
         // loop increments all of the iterators while leaving scalars unmodified
         // TODO this could be simplified with fold expressions or std::apply in C++17
         auto nop = [](auto &&...) {};
         for (auto itst = std::make_tuple(its...); GetNthElement<ColIdx>(its...) != end; nop(++its...)) {
            FillHistIt(itst, std::make_index_sequence<sizeof...(its)-1>{});
         }
      }

   public:
      using Result_t = HIST;

      FillBoostHelperAtomic(FillBoostHelperAtomic &&) = default;
      FillBoostHelperAtomic(const FillBoostHelperAtomic &) = delete;

      FillBoostHelperAtomic(HIST &&h) : fObject(std::make_shared<HIST>(std::move(h))) {
         static_assert(HIST::storage_type::has_threading_support);
      }

      void Initialize() {}

      void InitTask(TTreeReader *, unsigned int slot) {}



      // no container arguments
      template <typename... ValTypes,
               typename std::enable_if<!Disjunction<IsDataContainer<ValTypes>...>::value, int>::type = 0>
      void Exec(unsigned int slot, const ValTypes &...x) {
         using namespace boost::histogram;
         constexpr unsigned int N = sizeof...(x);

         const auto xst = std::forward_as_tuple(x...);

         FillHist(xst, std::make_index_sequence<N-1>{});
      }

      // at least one container argument
      template <typename... Xs, typename std::enable_if<Disjunction<IsDataContainer<Xs>...>::value, int>::type = 0>
      void Exec(unsigned int slot, const Xs &...xs)
      {
         // array of bools keeping track of which inputs are containers
         constexpr std::array<bool, sizeof...(Xs)> isContainer{IsDataContainer<Xs>::value...};

         // index of the first container input
         constexpr std::size_t colidx = FindIdxTrue(isContainer);
         // if this happens, there is a bug in the implementation
         static_assert(colidx < sizeof...(Xs), "Error: index of collection-type argument not found.");

         // get the end iterator to the first container
         auto const xrefend = std::end(GetNthElement<colidx>(xs...));

         // array of container sizes (1 for scalars)
         std::array<std::size_t, sizeof...(xs)> sizes = {{GetSize(xs)...}};

         for (std::size_t i = 0; i < sizeof...(xs); ++i) {
            if (isContainer[i] && sizes[i] != sizes[colidx]) {
               throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
            }
         }

         ExecLoop<colidx>(slot, xrefend, MakeBegin(xs)...);
      }

      void Finalize() {}

      std::shared_ptr<HIST> GetResultPtr() const {
         return fObject;
      }

      std::string GetActionName() { return "FillBoost"; }
   };

}


#endif

