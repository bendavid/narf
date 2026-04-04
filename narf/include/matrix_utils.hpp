#include <atomic>
#include <eigen3/Eigen/Dense>
#include <algorithm>

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

  std::size_t n_;
  std::vector<std::atomic<double> > data_;

};
