#pragma once

#include "concurrent_flat_map.hpp"
#include "histutils.hpp"
#include "matrix_utils.hpp"
#include "rdfutils.hpp"

#include <atomic>
#include <cmath>
#include <thread>
#include <unordered_set>
#include <vector>

namespace narf {
  ROOT::VecOps::RVec<double> testshift() {
    boost::histogram::axis::regular a(100, 0., 1.);
    boost::histogram::axis::category<std::string> b{"a", "b", "c"};
    HistShiftHelper<decltype(b), decltype(a)> helper(b, a);
    ROOT::VecOps::RVec<double> shifted0 {1e-3, 0.};
    ROOT::VecOps::RVec<std::string> shifted1 {"a", "b"};
    return helper(shifted1, shifted0, shifted1, shifted0, shifted1, shifted0);
  }
  
  using test_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2, 1>>;
  using vec_tensor_t = ROOT::VecOps::RVec<test_tensor_t>;
  
   vec_tensor_t testshifteigen() {
    boost::histogram::axis::regular a(100, 0., 1.);

    boost::histogram::axis::category<std::string> b{"a", "b", "c"};
    HistShiftHelper<decltype(b), decltype(a)> helper(b, a);
    test_tensor_t shifted0;
    shifted0(0, 0) = 1e-3;
    shifted0(1, 0) = 0.;
  
    std::string shifted1 = "a";
  
  
    ROOT::VecOps::RVec<std::string> vec1{"a", "a"};
    ROOT::VecOps::RVec<double> vec0{0., 0.};
  
    ROOT::VecOps::RVec<test_tensor_t> vecshifted0{shifted0, shifted0};
    return helper("a", vec0, shifted1, vecshifted0, shifted1, vecshifted0);
  }
  
  void testshiftrw() {
    std::vector<int> a{0, 1, 2};
    std::vector<int> b{3, 4, 5};

    std::cout << a.front() << std::endl;
    std::cout << b.front() << std::endl;

    for (auto const &[ael, bel] : make_zip_view(a, b)) {
      ael = 0;
      bel = 1;
    }

    std::cout << a.front() << std::endl;
    std::cout << b.front() << std::endl;


  }

  // Test SymMatrixAtomic: construction, fetch_add, fill_row, and index symmetry.
  // Returns true if all checks pass.
  bool testSymMatrixAtomic() {
    const std::size_t n = 4;
    SymMatrixAtomic mat(n);

    // Fill upper triangle: mat[i][j] = (i+1)*(j+1) for i <= j
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i; j < n; ++j) {
        mat.fetch_add(i, j, double((i + 1) * (j + 1)));
      }
    }

    // Verify via fill_row: rowData[j] == (i+1)*(j+1) for j >= i, else 0
    std::vector<double> rowData(n);
    for (std::size_t i = 0; i < n; ++i) {
      mat.fill_row(i, rowData.data());
      for (std::size_t j = 0; j < i; ++j) {
        if (rowData[j] != 0.0) return false;
      }
      for (std::size_t j = i; j < n; ++j) {
        if (rowData[j] != double((i + 1) * (j + 1))) return false;
      }
    }

    // Verify symmetry: fetch_add(i,j) and fetch_add(j,i) address the same element
    SymMatrixAtomic mat2(n);
    mat2.fetch_add(1, 3, 5.0);  // upper triangle (1,3)
    mat2.fetch_add(3, 1, 3.0);  // lower triangle: should map to same element
    mat2.fill_row(1, rowData.data());
    if (rowData[3] != 8.0) return false;

    return true;
  }

  // Test SparseMatrixAtomic: single-threaded fetch_add, index_values round-trip,
  // clear, and multi-threaded concurrent fetch_add. Returns true on success.
  bool testSparseMatrixAtomic() {
    // ---- single-threaded ----
    {
      SparseMatrixAtomic mat(20, 20);
      mat.fetch_add(1, 2, 3.0);
      mat.fetch_add(1, 2, 4.0);
      mat.fetch_add(5, 7, 1.5);
      mat.fetch_add(0, 0, 2.0);
      mat.fetch_add(0, 0, 0.0); // no-op
      if (mat(1, 2).load() != 7.0) return false;
      if (mat(5, 7).load() != 1.5) return false;
      if (mat(0, 0).load() != 2.0) return false;

      auto iv = mat.index_values();
      if (iv.size() != 3) return false;
      double sum = 0.0;
      for (std::size_t k = 0; k < iv.size(); ++k) sum += iv.vals()[k];
      if (sum != 10.5) return false;

      mat.clear();
      if (mat.index_values().size() != 0) return false;
      // reuse after clear
      mat.fetch_add(3, 4, 9.0);
      if (mat(3, 4).load() != 9.0) return false;
      if (mat.index_values().size() != 1) return false;
    }

    // ---- multi-threaded fetch_add: each (i,j) cell receives a known total ----
    {
      const std::size_t N = 32;
      SparseMatrixAtomic mat(N, N);
      const unsigned T = 8;
      const unsigned reps = 500;
      std::vector<std::thread> threads;
      threads.reserve(T);
      for (unsigned t = 0; t < T; ++t) {
        threads.emplace_back([&, t] {
          for (unsigned r = 0; r < reps; ++r) {
            for (std::size_t i = 0; i < N; ++i) {
              // Use a sparse pattern: only ~half the columns
              std::size_t j = (i * 3 + 1) % N;
              mat.fetch_add(i, j, 1.0 + 0.01 * t);
            }
          }
        });
      }
      for (auto& th : threads) th.join();

      double per_cell_expected = 0.0;
      for (unsigned t = 0; t < T; ++t) per_cell_expected += reps * (1.0 + 0.01 * t);

      auto iv = mat.index_values();
      if (iv.size() != N) return false;
      for (std::size_t k = 0; k < iv.size(); ++k) {
        std::size_t i = iv.idxs0()[k];
        std::size_t j = iv.idxs1()[k];
        if (j != (i * 3 + 1) % N) return false;
        if (std::abs(iv.vals()[k] - per_cell_expected) > 1e-9) return false;
      }
    }

    return true;
  }

  // Test concurrent_flat_map: single-threaded correctness, expansion, and
  // multi-threaded concurrent insert / find. Returns true on success.
  bool testConcurrentFlatMap() {
    // ---- single-threaded correctness, force several expansions ----
    {
      concurrent_flat_map<std::uint64_t, std::uint64_t> map(8);
      const std::uint64_t N = 5000;
      for (std::uint64_t i = 1; i <= N; ++i) {
        auto [p, inserted] = map.emplace(i, i * 7u + 3u);
        if (!inserted || !p || *p != i * 7u + 3u) return false;
      }
      // re-insert: must report not-inserted but return existing value
      for (std::uint64_t i = 1; i <= N; ++i) {
        auto [p, inserted] = map.emplace(i, std::uint64_t(0));
        if (inserted || !p || *p != i * 7u + 3u) return false;
      }
      // find every key
      for (std::uint64_t i = 1; i <= N; ++i) {
        auto* p = map.find(i);
        if (!p || *p != i * 7u + 3u) return false;
      }
      // missing keys
      if (map.find(0) != nullptr) return false;
      if (map.find(N + 1) != nullptr) return false;
    }

    // ---- pointer stability across expansion ----
    {
      concurrent_flat_map<std::uint64_t, std::uint64_t> map(4);
      std::vector<std::uint64_t*> ptrs;
      const std::uint64_t N = 1000;
      ptrs.reserve(N);
      for (std::uint64_t i = 1; i <= N; ++i) {
        ptrs.push_back(map.emplace(i, i).first);
      }
      for (std::uint64_t i = 1; i <= N; ++i) {
        if (ptrs[i - 1] != map.find(i)) return false;
        if (*ptrs[i - 1] != i) return false;
      }
    }

    // ---- multi-threaded insert / find ----
    {
      concurrent_flat_map<std::uint64_t, std::uint64_t> map(16);
      const unsigned T = 8;
      const std::uint64_t per = 4000;
      std::atomic<std::uint64_t> dup_inserts{0};
      std::atomic<std::uint64_t> bad{0};
      std::vector<std::thread> threads;
      threads.reserve(T);
      for (unsigned t = 0; t < T; ++t) {
        threads.emplace_back([&, t] {
          for (std::uint64_t i = 0; i < per; ++i) {
            // Overlapping key ranges across threads exercise contention.
            std::uint64_t key = (i % (per / 2)) + 1 + (t % 2) * (per / 2);
            std::uint64_t val = key * 1315423911u;
            auto [p, ins] = map.emplace(key, val);
            if (!p || *p != val) bad.fetch_add(1);
            (void)ins;
            auto* f = map.find(key);
            if (!f || *f != val) bad.fetch_add(1);
          }
          // Each thread also inserts a unique key block.
          for (std::uint64_t i = 0; i < per; ++i) {
            std::uint64_t key = 1000000ull + t * per + i;
            auto [p, ins] = map.emplace(key, key ^ 0xdeadbeefu);
            if (!ins || !p || *p != (key ^ 0xdeadbeefu)) {
              dup_inserts.fetch_add(1);
            }
          }
        });
      }
      for (auto& th : threads) th.join();
      if (bad.load() != 0) return false;
      if (dup_inserts.load() != 0) return false;

      // Verify all unique-block keys present and correct.
      for (unsigned t = 0; t < T; ++t) {
        for (std::uint64_t i = 0; i < per; ++i) {
          std::uint64_t key = 1000000ull + t * per + i;
          auto* p = map.find(key);
          if (!p || *p != (key ^ 0xdeadbeefu)) return false;
        }
      }
    }

    return true;
  }

  // Test QuantileHelperStatic: edges {0.25, 0.5, 0.75} partition into 4 bins.
  // Exercise both scalar and container (RVec) call paths via MapWrapper.
  bool testQuantileHelperStatic() {
    QuantileHelperStatic<4> helper(std::array<double, 4>{0.25, 0.5, 0.75, 1.0});

    if (helper(0.1) != 0) return false;
    if (helper(0.25) != 1) return false;
    if (helper(0.4) != 1) return false;
    if (helper(0.6) != 2) return false;
    if (helper(0.9) != 3) return false;

    ROOT::VecOps::RVec<double> vals{0.1, 0.4, 0.6, 0.9};
    auto res = helper(vals);
    if (res.size() != 4) return false;
    if (res[0] != 0 || res[1] != 1 || res[2] != 2 || res[3] != 3) return false;

    // Continuous mode: CDF in [0, 1], edges[i] -> i/(N-1).
    QuantileHelperStaticContinuous<4> helper_c(std::array<double, 4>{0.25, 0.5, 0.75, 1.0});
    auto const eps = 1e-9;
    if (std::abs(helper_c(0.25) - 0.0) > eps) return false;
    if (std::abs(helper_c(0.5) - 1.0/3.0) > eps) return false;
    if (std::abs(helper_c(1.0) - 1.0) > eps) return false;
    // val below first edge clamps to 0
    if (std::abs(helper_c(0.0) - 0.0) > eps) return false;
    // val above last edge clamps to 1
    if (std::abs(helper_c(2.0) - 1.0) > eps) return false;
    // midpoint of first interval
    if (std::abs(helper_c(0.375) - (1.0/6.0)) > eps) return false;

    // Container path through MapWrapper
    auto res_c = helper_c(ROOT::VecOps::RVec<double>{0.25, 0.5, 0.75, 1.0});
    if (res_c.size() != 4) return false;
    if (std::abs(res_c[0] - 0.0) > eps) return false;
    if (std::abs(res_c[1] - 1.0/3.0) > eps) return false;
    if (std::abs(res_c[2] - 2.0/3.0) > eps) return false;
    if (std::abs(res_c[3] - 1.0) > eps) return false;

    return true;
  }

  // Test MapWrapper: constructs a wrapper over a simple callable and applies
  // it over zipped input ranges.
  bool testMapWrapper() {
    auto add = [](int a, int b) { return a + b; };
    MapWrapper<decltype(add)> wrapper(add);

    // Scalar passthrough: no container args -> callable invoked directly.
    auto add_scalar = [](int x, int y) { return x + y; };
    MapWrapper<decltype(add_scalar)> scalar_wrapper(add_scalar);
    if (scalar_wrapper(2, 5) != 7) return false;

    std::vector<int> a{1, 2, 3, 4};
    std::vector<int> b{10, 20, 30, 40};
    auto res = wrapper(a, b);
    if (res.size() != 4) return false;
    if (res[0] != 11) return false;
    if (res[1] != 22) return false;
    if (res[2] != 33) return false;
    if (res[3] != 44) return false;
    return true;
  }
}
