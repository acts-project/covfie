if __name__ == "__main__":
    for n in range(1, 11):
        print(f"template <typename F, std::enable_if_t<utility::backend_depth<typename F::backend_t>::value == {n}, bool> = true> auto make_parameter_pack_for(")
        print(",".join(f"typename utility::nth_backend<typename F::backend_t, {i}>::type::configuration_t && a{i}" for i in range(n)))
        print(") { return make_parameter_pack(")
        print(",".join(f"std::forward<typename utility::nth_backend<typename F::backend_t, {i}>::type::configuration_t>(a{i})" for i in range(n)))
        print(");}")
        print()
