#pragma once
// eigen_tensor_type_deduction.hpp
//
// C++20 utility that inspects an arbitrary Eigen Tensor expression and
// deduces the concrete Eigen::Tensor<…> or Eigen::TensorFixedSize<…>
// type needed to materialise (evaluate / assign) that expression.
//
// For types that are already concrete tensors (Tensor, TensorFixedSize)
// or are not Eigen tensor expressions at all, both tensor_result_t and
// tensor_eval pass the original type and reference straight through
// without materialisation.
//
// ────────────────────────────────────────────────────────────────────
// Usage:
//
//   Eigen::Tensor<float, 3> a(2, 3, 4);
//   auto expr = a + a * 2.f;
//
//   // 1. Deduce the concrete type for an expression:
//   using ResultType = tensor_result_t<decltype(expr)>;
//   //  => Eigen::Tensor<float, 3, ColMajor, Eigen::DenseIndex>
//
//   // 2. Evaluate the expression:
//   auto result = tensor_eval(expr);
//
//   // 3. Pass‑through for already‑concrete tensors (no copy):
//   using Same = tensor_result_t<Eigen::Tensor<float, 3>>;
//   //  => Eigen::Tensor<float, 3, …>  (the exact same type)
//   auto& ref = tensor_eval(a);  // returns by reference, no copy
//
//   // 4. Pass‑through for non‑Eigen types:
//   using Int = tensor_result_t<int>;   //  => int
//   int x = 42;
//   auto& r = tensor_eval(x);          // forwards reference
//
//   // 5. Fixed‑size expressions:
//   Eigen::TensorFixedSize<double, Eigen::Sizes<2,3>> fs;
//   using FSR = tensor_result_t<decltype(fs + fs)>;
//   //  => Eigen::TensorFixedSize<double, Sizes<2,3>, ColMajor, …>
//
// ────────────────────────────────────────────────────────────────────

#include <type_traits>
#include <cstddef>
#include <utility>

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace narf {

namespace eigen_tensor_detail {

// ─── Detect Eigen::Sizes<…> (compile‑time fixed dimensions) ────────

template <typename T>
struct is_sizes : std::false_type {};

template <std::ptrdiff_t... Dims>
struct is_sizes<Eigen::Sizes<Dims...>> : std::true_type {};

template <typename T>
inline constexpr bool is_sizes_v = is_sizes<T>::value;

// ─── Detect concrete Eigen tensor storage types ─────────────────────

template <typename T>
using bare_t = std::remove_cvref_t<T>;

template <typename T> struct is_tensor : std::false_type {};
template <typename S, int N, int O, typename I>
struct is_tensor<Eigen::Tensor<S, N, O, I>> : std::true_type {};

template <typename T> struct is_tensor_fixed_size : std::false_type {};
template <typename S, typename D, int O, typename I>
struct is_tensor_fixed_size<Eigen::TensorFixedSize<S, D, O, I>> : std::true_type {};

/// True if T is Eigen::Tensor<…> or Eigen::TensorFixedSize<…>.
template <typename T>
inline constexpr bool is_concrete_tensor_v =
    is_tensor<bare_t<T>>::value || is_tensor_fixed_size<bare_t<T>>::value;

// ─── Detect any Eigen tensor expression ─────────────────────────────
//
// Every Eigen tensor type (concrete or expression node) has a
// specialisation of Eigen::internal::traits with NumDimensions.
// Non‑Eigen types do not.

template <typename T, typename = void>
struct is_eigen_tensor_expr : std::false_type {};

template <typename T>
struct is_eigen_tensor_expr<T,
    std::void_t<decltype(Eigen::internal::traits<bare_t<T>>::NumDimensions)>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_eigen_tensor_expr_v = is_eigen_tensor_expr<T>::value;

/// True if T is an Eigen tensor expression that is NOT already a
/// concrete Tensor or TensorFixedSize (i.e. it needs materialisation).
template <typename T>
inline constexpr bool needs_eval_v =
    is_eigen_tensor_expr_v<T> && !is_concrete_tensor_v<T>;

// ─── Trait extraction (only instantiated for Eigen expressions) ─────

template <typename E>
using scalar_t = typename Eigen::internal::traits<bare_t<E>>::Scalar;

template <typename E>
using index_t = typename Eigen::internal::traits<bare_t<E>>::Index;

template <typename E>
inline constexpr int layout_v =
    static_cast<int>(Eigen::internal::traits<bare_t<E>>::Layout);

template <typename E>
inline constexpr int rank_v =
    static_cast<int>(Eigen::internal::traits<bare_t<E>>::NumDimensions);

// ─── Evaluator‑based dimension type detection ──────────────────────
//
// TensorEvaluator<Expr, DefaultDevice>::Dimensions resolves to:
//   • Eigen::Sizes<d0, d1, …>  when compile‑time dimensions survive
//   • Eigen::DSizes<Index, N>   otherwise (dynamic)

template <typename E>
using eval_dims_t = typename Eigen::TensorEvaluator<
    std::add_const_t<bare_t<E>>,
    Eigen::DefaultDevice>::Dimensions;

template <typename E>
inline constexpr bool has_fixed_sizes_v = is_sizes_v<eval_dims_t<E>>;

// ─── Expression → concrete type deduction ───────────────────────────

template <typename E, bool Fixed = has_fixed_sizes_v<E>>
struct deduce_tensor_result;

template <typename E>
struct deduce_tensor_result<E, false>
{
    using type = Eigen::Tensor<
        scalar_t<E>,
        rank_v<E>,
        layout_v<E>,
        index_t<E>>;
};

template <typename E>
struct deduce_tensor_result<E, true>
{
    using type = Eigen::TensorFixedSize<
        scalar_t<E>,
        eval_dims_t<E>,
        layout_v<E>,
        index_t<E>>;
};

// ─── Top‑level result‑type dispatch ─────────────────────────────────
//
// Three cases:
//   1. Needs evaluation (expression node)  →  deduce concrete type
//   2. Already concrete tensor             →  pass through as‑is
//   3. Not an Eigen type at all            →  pass through as‑is

template <typename T, typename = void>
struct tensor_result
{
    // Case 2 & 3: pass through
    using type = bare_t<T>;
};

template <typename T>
struct tensor_result<T, std::enable_if_t<needs_eval_v<T>>>
{
    // Case 1: deduce
    using type = typename deduce_tensor_result<T>::type;
};

} // namespace eigen_tensor_detail

// ═══════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════

/// Type alias: the concrete Tensor or TensorFixedSize type that can
/// store the result of the expression `E`.
///
/// For already‑concrete tensors and non‑Eigen types, this is just the
/// (unqualified) type itself.
template <typename E>
using tensor_result_t =
    typename eigen_tensor_detail::tensor_result<
        std::remove_cvref_t<E>>::type;

/// Evaluate an Eigen tensor expression into a concrete tensor value.
///
/// For types that are already concrete tensors or are not Eigen types
/// at all, this perfectly forwards the reference (no copy).
///
/// Overload 1: expression → materialise into a new concrete tensor.
template <typename E>
    requires eigen_tensor_detail::needs_eval_v<std::remove_cvref_t<E>>
[[nodiscard]] auto tensor_eval(E&& expr) -> tensor_result_t<E>
{
    tensor_result_t<E> result = std::forward<E>(expr);
    return result;
}

/// Overload 2: already concrete / non‑Eigen → perfect‑forward.
template <typename E>
    requires (!eigen_tensor_detail::needs_eval_v<std::remove_cvref_t<E>>)
[[nodiscard]] constexpr decltype(auto) tensor_eval(E&& val) noexcept
{
    return std::forward<E>(val);
}

// ═══════════════════════════════════════════════════════════════════════
// Compile‑time introspection helpers
// ═══════════════════════════════════════════════════════════════════════

/// True if T is an Eigen tensor expression (concrete or lazy).
template <typename E>
inline constexpr bool is_eigen_tensor_v =
    eigen_tensor_detail::is_eigen_tensor_expr_v<std::remove_cvref_t<E>>;

/// True if T is an Eigen expression that requires materialisation
/// (i.e. not already Tensor or TensorFixedSize).
template <typename E>
inline constexpr bool is_tensor_expression_v =
    eigen_tensor_detail::needs_eval_v<std::remove_cvref_t<E>>;

/// True if T is an Eigen::Tensor<…> or Eigen::TensorFixedSize<…>.
template <typename E>
inline constexpr bool is_concrete_tensor_v =
    eigen_tensor_detail::is_concrete_tensor_v<std::remove_cvref_t<E>>;

/// True if the result of expression E would be a TensorFixedSize.
/// Only meaningful for Eigen tensor types.
template <typename E>
    requires is_eigen_tensor_v<E>
inline constexpr bool is_fixed_size_expr_v =
    eigen_tensor_detail::has_fixed_sizes_v<std::remove_cvref_t<E>>;

/// Number of dimensions of the expression.
/// Only meaningful for Eigen tensor types.
template <typename E>
    requires is_eigen_tensor_v<E>
inline constexpr int tensor_expr_rank_v =
    eigen_tensor_detail::rank_v<std::remove_cvref_t<E>>;

/// Scalar type of the expression.
/// Only meaningful for Eigen tensor types.
template <typename E>
    requires is_eigen_tensor_v<E>
using tensor_expr_scalar_t =
    eigen_tensor_detail::scalar_t<std::remove_cvref_t<E>>;

}
