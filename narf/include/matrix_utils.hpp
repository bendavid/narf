#pragma once

#include <atomic>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include "concurrent_flat_map.hpp"


class SymMatrixAtomic {
public:
  SymMatrixAtomic() = default;
  SymMatrixAtomic(std::size_t n) : n_(n), data_(n*(n+1)/2) {}

  double fetch_add(std::size_t iidx, std::size_t jidx, double val) {
    std::atomic<double>& ref = data_[packed_index(iidx, jidx)];
    return ref.fetch_add(val);
  }

  void fill_row(std::size_t row, double *rowData) {
    const std::size_t offset = packed_index(row, row);
    std::fill(rowData, rowData + row, 0.);
    std::copy(data_.begin() + offset, data_.begin() + offset + n_ - row, rowData + row);
  }

private:

/**
 * Converts (row, col) indices of a symmetric matrix into a linearized index
 * for packed storage of unique elements (upper triangle, row-major).
 *
 * Storage layout (0-indexed, n=4 example):
 *   (0,0)(0,1)(0,2)(0,3) | (1,1)(1,2)(1,3) | (2,2)(2,3) | (3,3)
 *     0    1    2    3       4    5    6       7    8       9
 *
 * Total elements stored: n*(n+1)/2
 *
 * @param row  Row index (0-based)
 * @param col  Column index (0-based)
 * @return     Linearized index into packed storage array
 */
    inline std::size_t packed_index(std::size_t row, std::size_t col) {
        // Normalize to upper triangle: ensure row <= col
        if (row > col) {
            std::size_t tmp = row;
            row = col;
            col = tmp;
        }

        // Number of elements in rows 0..(row-1): row*n - row*(row-1)/2
        // Plus offset within current row: (col - row)
        return row * n_ - row * (row - 1) / 2 + (col - row);
    }

  std::size_t n_{};
  std::vector<std::atomic<double> > data_;

};

class SparseMatrixIndexValues {
public:
  const std::vector<std::size_t> &idxs0() const { return idxs0_; }
  const std::vector<std::size_t> &idxs1() const { return idxs1_; }
  const std::vector<double> &vals() const { return vals_; }

  void emplace_back(std::size_t idx0, std::size_t idx1, double val) {
    idxs0_.emplace_back(idx0);
    idxs1_.emplace_back(idx1);
    vals_.emplace_back(val);
  }

  void reserve(std::size_t i) {
    idxs0_.reserve(i);
    idxs1_.reserve(i);
    vals_.reserve(i);
  }

  std::size_t size() const { return vals_.size(); }

private:
  std::vector<std::size_t> idxs0_;
  std::vector<std::size_t> idxs1_;
  std::vector<double> vals_;
};

class SparseMatrixAtomic {
public:
  using map_type = narf::concurrent_flat_map<std::size_t, std::atomic<double>>;

  SparseMatrixAtomic(std::size_t size0, std::size_t size1)
    : size0_(size0), size1_(size1),
      data_(std::max<std::size_t>(size0 * size1 / 40, 16)) {}

  std::atomic<double> &operator() (std::size_t idx0, std::size_t idx1) {
    const std::size_t i = globalidx(idx0, idx1);
    auto res = data_.emplace(i);
    return *res.first;
  }

  const std::atomic<double> &operator() (std::size_t idx0, std::size_t idx1) const {
    const std::size_t i = globalidx(idx0, idx1);
    auto* p = data_.find(i);
    return *p;
  }

  void fetch_add(std::size_t idx0, std::size_t idx1, double val) {
    if (val != 0.) {
      auto &elemval = operator()(idx0, idx1);
      elemval.fetch_add(val);
    }
  }

  SparseMatrixIndexValues index_values() const {
    SparseMatrixIndexValues res;
    res.reserve(data_.size());

    data_.for_each([&](std::size_t key, const std::atomic<double>& val) {
      auto is = idxs(key);
      res.emplace_back(is[0], is[1], val.load());
    });

    return res;
  }

  void clear() { data_.clear(); }

  void reserve(std::size_t /*i*/) { /* no-op: map grows on demand */ }

  std::size_t dense_size() const { return size0_*size1_; }

  map_type &data() { return data_; }

private:
  std::size_t globalidx(std::size_t idx0, std::size_t idx1) const {
    return idx0*size0_ + idx1;
  }

  std::array<std::size_t, 2> idxs(std::size_t globalidx) const {
    const std::size_t idx0 = globalidx/size0_;
    const std::size_t idx1 = globalidx % size0_;

    return std::array<std::size_t, 2>{idx0, idx1};
  }

  const std::size_t size0_;
  const std::size_t size1_;
  map_type data_;

};