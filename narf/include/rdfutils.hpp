#pragma once

#include "tensorevalutils.hpp"
#include "utils.hpp"

namespace narf {

    template<typename Callable, typename... Args>
    class DefineWrapper {

    private:
        Callable callable_;

        using return_t = std::decay_t<decltype(callable_(std::declval<Args>()...))>;

    public:

        DefineWrapper(const Callable &callable) : callable_(callable) {}
        DefineWrapper(Callable &&callable) : callable_(std::move(callable)) {}

        return_t operator() (const Args&... args) {
            return callable_(args...);
        }
    };

    template<typename Callable>
    class MapWrapper {

    private:
        Callable callable_;
    public:

        template<typename... CArgs>
            requires std::is_constructible_v<Callable, CArgs&&...>
        explicit MapWrapper(CArgs&&... cargs) : callable_(std::forward<CArgs>(cargs)...) {}

        template<typename... Args>
        auto operator() (const Args&... args) {
            if constexpr ((is_container<Args> || ...)) {
                auto apply_elem = [this](auto const &elem_tuple) {
                    return std::apply(callable_, elem_tuple);
                };
                auto const res_view = make_zip_view(make_view(args)...) | std::views::transform(apply_elem);
                return range_to(res_view);
            } else {
                return callable_(args...);
            }
        }
    };

    // TensorMapWrapper: applies the wrapped callable element-wise over
    // Eigen tensor arguments.  Exactly one argument may be a concrete
    // Eigen tensor; all other arguments are broadcast (passed through
    // unchanged to every element call).  The result is a tensor of the
    // same shape whose scalar type is the callable's return type.
    // When no tensor argument is present, calls through directly.
    template<typename Callable>
    class TensorMapWrapper {

    private:
        Callable callable_;

        // View adapter: tensor args produce a finite view over their
        // elements; everything else repeats (same pattern as MapWrapper's
        // make_view, but keyed on is_concrete_tensor_v).
        template <typename T>
        static auto make_tensor_view(const T &x) {
            if constexpr (is_concrete_tensor_v<T>) {
                return std::views::counted(x.data(), x.size());
            } else {
                return make_repeat_view(x);
            }
        }

        template <typename... Args>
        static consteval std::size_t find_tensor_idx() {
            constexpr bool flags[] = {is_concrete_tensor_v<Args>...};
            for (std::size_t i = 0; i < sizeof...(Args); ++i)
                if (flags[i]) return i;
            return sizeof...(Args);
        }

    public:

        template<typename... CArgs>
            requires std::is_constructible_v<Callable, CArgs&&...>
        explicit TensorMapWrapper(CArgs&&... cargs)
            : callable_(std::forward<CArgs>(cargs)...) {}

        template<typename... Args>
        auto operator() (const Args&... args) {
            if constexpr ((is_concrete_tensor_v<Args> || ...)) {
                auto apply_elem = [this](auto const &elem_tuple) {
                    return std::apply(callable_, elem_tuple);
                };
                auto const res_view =
                    make_zip_view(make_tensor_view(args)...)
                    | std::views::transform(apply_elem);

                // Deduce output tensor type from the input tensor's
                // shape and the callable's return type.
                constexpr std::size_t TIdx = find_tensor_idx<Args...>();
                auto const &tensor_arg =
                    std::get<TIdx>(std::tie(args...));
                using in_tensor_t =
                    std::remove_cvref_t<decltype(tensor_arg)>;
                using out_scalar_t =
                    std::decay_t<decltype(*res_view.begin())>;

                Eigen::TensorFixedSize<out_scalar_t,
                                       typename in_tensor_t::Dimensions>
                    result;
                auto it = res_view.begin();
                for (Eigen::Index k = 0; k < result.size(); ++k, ++it)
                    result.data()[k] = *it;
                return result;
            } else {
                return callable_(args...);
            }
        }
    };

}
