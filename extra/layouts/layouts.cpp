#include <array>

#include <x86intrin.h>

template <std::size_t N>
std::size_t morton_bitshift(std::array<std::size_t, N> c)
{
    std::size_t r = 0;

    for (std::size_t i = 0; i < (64 / N); ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            r |= (c[j] & (1UL << i)) << (i * (N - 1) + j);
        }
    }

    return r;
}

static constexpr std::size_t get_mask(std::size_t n, std::size_t i)
{
    std::size_t r = 0;

    for (std::size_t i = 0; i < 64; ++i) {
        r |= (i % n == 0 ? (1UL << i) : 0UL);
    }

    return r << i;
}

template <std::size_t N>
std::size_t morton_pdep(std::array<std::size_t, N> c)
{

    std::size_t r = 0;
    for (std::size_t i = 0; i < N; ++i) {
        r |= _pdep_u64(c[i], get_mask(N, i));
    }
    return r;
}

template <std::size_t N>
std::size_t
pitched_naive(std::array<std::size_t, N> c, std::array<std::size_t, N> s)
{
    std::size_t r = 0;

    for (std::size_t k = 0; k < N; ++k) {
        std::size_t tmp = c[k];

        for (std::size_t l = k + 1; l < N; ++l) {
            tmp *= s[l];
        }

        r += tmp;
    }

    return r;
}

template <std::size_t N>
std::size_t
pitched_fast(std::array<std::size_t, N> c, std::array<std::size_t, N> s)
{
    std::size_t r = c[N - 1];

    for (std::size_t i = N - 2; i < N; --i) {
        r = (r * s[i]) + c[i];
    }

    return r;
}

template <std::size_t N>
std::size_t
pitched_precalc(std::array<std::size_t, N> c, std::array<std::size_t, N> s)
{
    std::size_t r = 0;

    for (std::size_t k = 0; k < N; ++k) {
        r += c[k] * s[k];
    }

    return r;
}

#if defined(BUILD_MORTONBITSHIFT)
template std::size_t morton_bitshift(std::array<std::size_t, NDIMS> c);
#elif defined(BUILD_MORTONPDEP)
template std::size_t morton_pdep(std::array<std::size_t, NDIMS> c);
#elif defined(BUILD_PITCHEDNAIVE)
template std::size_t pitched_naive(
    std::array<std::size_t, NDIMS> c, std::array<std::size_t, NDIMS> s
);
#elif defined(BUILD_PITCHEDFAST)
template std::size_t pitched_fast(
    std::array<std::size_t, NDIMS> c, std::array<std::size_t, NDIMS> s
);
#elif defined(BUILD_PITCHEDPRECALC)
template std::size_t pitched_precalc(
    std::array<std::size_t, NDIMS> c, std::array<std::size_t, NDIMS> s
);
#endif
