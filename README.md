# covfie

![](https://github.com/stephenswat/covfie/actions/workflows/builds.yml/badge.svg?branch=main)
![](https://github.com/stephenswat/covfie/actions/workflows/checks.yml/badge.svg?branch=main)
[![Documentation Status](https://readthedocs.org/projects/covfie/badge/?version=latest)](https://covfie.readthedocs.io/en/latest/?badge=latest)

**covfie** (pronounced _coffee_) is a **co**-processor **v**ector **fie**ld
library. covfie consists of two main components; the first is the header-only
C++ library, which can be used by scientific applications using CUDA or other
programming platforms. The second is a set of benchmarks which can be used to
quantify the computational performance of all the different vector field
implementations covfie provides. Arguably, the test suite constitutes a third
component.

## Quick start

All covfie vector fields are stored in the `covfie::field` type and they are
accessed using the `covfie::field_view` type. These require a backend type
passed as a template parameters, and their behaviour can be expanded using
transformers. The easiest way to get started using covfie is to use a field
builder backend, which allows you to write data to memory (the following
example is included in the repository as
[`readme_example_1`](examples/core/readme_example_1.cpp)):

```cpp
using field_t = covfie::field<covfie::backend::builder<
    covfie::backend::vector::input::ulong2,
    covfie::backend::vector::output::float2>>;

int main(void)
{
    // Initialize the field as a 10x10 field, then create a view from it.
    field_t my_field(field_t::backend_t::configuration_data_t{10ul, 10ul});
    field_t::view_t my_view(my_field);

    // Assign f(x, y) = (sin x, cos y)
    for (std::size_t x = 0ul; x < 10ul; ++x) {
        for (std::size_t y = 0ul; y < 10ul; ++y) {
            my_view.at(x, y)[0] = std::sin(static_cast<float>(x));
            my_view.at(x, y)[1] = std::cos(static_cast<float>(y));
        }
    }

    // Retrieve the vector value at (2, 3)
    field_t::output_t v = my_view.at(2ul, 3ul);

    std::cout << "Value at (2, 3) = (" << v[0] << ", " << v[1] << ")"
              << std::endl;

    return 0;
}
```

This next example ([`readme_example_2`](examples/core/readme_example_2.cpp))
creates a two-dimensional vector field over the natural numbers, stretching 10
indices in each direction. If we want to use real numbers for our vector field,
we can simply add a linear interpolator:

```cpp
using builder_t = covfie::field<covfie::backend::builder<
    covfie::backend::vector::input::ulong2,
    covfie::backend::vector::output::float2>>;

using field_t =
    covfie::field<covfie::backend::transformer::interpolator::linear<
        covfie::backend::layout::strided<
            covfie::backend::vector::input::ulong2,
            covfie::backend::storage::array<
                covfie::backend::vector::output::float2>>>>;

int main(void)
{
    // Initialize the field as a 10x10 field, then create a view from it.
    builder_t my_field(builder_t::backend_t::configuration_data_t{10ul, 10ul});
    builder_t::view_t my_view(my_field);

    // Assign f(x, y) = (sin x, cos y)
    for (std::size_t x = 0ul; x < 10ul; ++x) {
        for (std::size_t y = 0ul; y < 10ul; ++y) {
            my_view.at(x, y)[0] = std::sin(static_cast<float>(x));
            my_view.at(x, y)[1] = std::cos(static_cast<float>(y));
        }
    }

    field_t new_field(my_field);
    field_t::view_t new_view(new_field);

    // Retrieve the vector value at (2.31, 3.98)
    field_t::output_t v = new_view.at(2.31f, 3.98f);

    std::cout << "Value at (2.31, 3.98) = (" << v[0] << ", " << v[1] << ")"
              << std::endl;

    return 0;
}
```

covfie types can seem intimidating at first, but they are quite friendly! Also,
you only really need to worry about them once, and you can hide them away in a
typedef.

## Dependencies

covfie is light on dependencies. In fact, it doesn't have any at all. However,
if you want to build the examples or the tests you should have `boost`
installed. The tests are based on `googletest`, and the CUDA components require
the NVIDIA CUDA toolkit to be available.

## Building

Building covfie is done using CMake. The following is an example of how you may
choose to build the code:

```bash
$ cmake -B build -S [source] -DCOVFIE_BUILD_EXAMPLES=On -DCOVFIE_PLATFORM_CPU=On
$ cmake --build build -- -j 4
```

Please note that &ndash; because covfie is a header only library &ndash;
nothing actually needs to be compiled if you don't want the examples or the
tests. The following is a list of flags that can be used to configure the
build:

| Flag | Meaning |
| - | - |
| `COVFIE_BUILD_EXAMPLES` | Build the examples. |
| `COVFIE_BUILD_TESTS` | Build the test executables. |
| `COVFIE_BUILD_BENCHMARKS` | Build the benchmarks. |
| `COVFIE_PLATFORM_CPU` | Build CPU-specific code. |
| `COVFIE_PLATFORM_CUDA` | Build CUDA-specific code. |
| `COVFIE_REQUIRE_CXX20` | Emit an error instead of a warning if support for C++20 concepts is missing. |
