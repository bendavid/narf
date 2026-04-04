#pragma once

#include "histutils.hpp"
#include "matrix_utils.hpp"

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
}
