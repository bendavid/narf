#pragma once

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

        MapWrapper(const Callable &callable) : callable_(callable) {}
        MapWrapper(Callable &&callable) : callable_(std::move(callable)) {}

        template<typename... CArgs>
            requires (sizeof...(CArgs) != 1 || (!std::is_same_v<std::decay_t<CArgs>, MapWrapper> && ...)) &&
                     std::is_constructible_v<Callable, CArgs&&...>
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

}
