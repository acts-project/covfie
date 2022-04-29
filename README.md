# Covfie

![](https://github.com/stephenswat/covfie/actions/workflows/builds.yml/badge.svg?branch=main)
![](https://github.com/stephenswat/covfie/actions/workflows/checks.yml/badge.svg?branch=main)

**covfie** (pronounced _coffee_) is a **co**-processor **v**ector **fie**ld
projects. Covfie consists of two main components; the first is the library
itself, which can be used by scientific applications using CUDA or other
programming platforms. The second is a set of benchmarks which can be used to
quantify the computational performance of all the different vector field
implementations. Arguably, the test suite forms a third component. This README
will be concerned mostly with the library part of the project.

## Core ideas

The library part of Covfie is a header-only C++ library. In Covfie, we aim to
model uniformly sampled vector fields ℝ<sup>n</sup> → ℝ<sup>m</sup>. The
problem is that the memory inside (most computers) can be described &ndash; at
best &ndash; as an ℕ<sup>1</sup> → ℝ<sup>1</sup> mapping; we have quite a long
way to go to get to our goal of representing more complicated vector fields.

In covfie, we tackle this complexity through composition. In other words, we
break the problem into the tiniest possible pieces and then put them together
in a clever way, re-using pieces along the way. This allows us to minimize the
total amount of code we need to perform the computations we want, and it means
that you never pay (in performance) for functionality you don't need. If
something does not benefit you, you simply do not add it to the vector field
which you are constructing.

covfie is built on the following design principles:

1. Know everything at compile time
    * Anything we can't know at compile time can come back to haunt us later
    * The compiler is our friend, leverage it to find bugs
    * Allow maximum optimisation by showing the compiler everything
    * Absolutely no run-time polymorphism
2. Composition is **the** (only) way to control complexity
    * As Brian Beckman said: modern software is to complex by any other means
    * Factor code into as many small pieces as possible
    * If you're repeating yourself, you're not abstract enough
    * Start from simple building blocks
3. Be ergonomic at the point of use
    * The backend may be a mess, but using covfie should be easy
    * Assume the user has fixed most of their field parameters already
    * Provide a comfortable, easy to remember API
4. Extensibility is key
    * Code should be extensible on the user's end
    * Provide easy-to-use kinds that user types and type constructors can slot
      into

## Quick start

All covfie vector fields are stored in the `covfie::field` type and they are
accessed using the `covfie::field_view` type. These require a backend type
passed as a template parameters, and their behaviour can be expanded using
transformers. The easiest way to get started using covfie is to use a field
builder backend, which allows you to write data to memory:

```cpp
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/core/backend/builder.hpp>

using field_t = covfie::field<covfie::backend::builder<3, 3>>;

field_t my_field(field_t::backend_t::configuration_data_t{10u, 10u, 10u});
field_t::view_t my_view(my_field);

my_view.at(1u, 5u, 4u)[0] = 4.12f;
```

This code creates a three-dimensional vector field over the natural numbers,
stretching 10 indices in each direction. If we want to use real numbers for our
vector field, we can simply add a nearest neighbour interpolator:

```cpp
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/core/backend/builder.hpp>
#include <covfie/core/backend/transformer/interpolator/nearest_neighbour.hpp>

using field_t =
    covfie::field<
        covfie::backend::transformer::interpolator::nearest_neighbour<
            covfie::backend::builder<3, 3>
        >
    >;

field_t my_field(
    field_t::backend_t::configuration_data_t{}
    field_t::backend_t::backend_t::configuration_data_t{10u, 10u, 10u},
);
field_t::view_t my_view(my_field);

my_view.at(4.2f, 1.2f, -11.9f)[0] = 4.12f;
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
