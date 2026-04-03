#pragma once

namespace narf {

    template<typename Callable, typename... Args>
    class DefineWrapper {

    private:
        Callable callable_;

        using return_t = decltype(callable_(std::declval<Args>()...));

    public:

        DefineWrapper(const Callable &callable) : callable_(callable) {}
        DefineWrapper(Callable &&callable) : callable_(std::move(callable)) {}

        return_t operator() (const Args&... args) const {
            return callable_(args...);
        }
    };
}
