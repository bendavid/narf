#pragma once

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
}
