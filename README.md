# Covfie

![](https://github.com/stephenswat/covfie/actions/workflows/builds.yml/badge.svg?branch=main)

Covfie (pronounced _coffee_) is a **Co**-processor **v**ector **fie**ld
library. Covfie consists of two main components; the first is the library
itself, which can be used by any scientific application using CUDA or other
programming platforms. The second is a set of benchmarks which can be used to
quantify the computational performance of all the different vector field
implementations.

## The library

The library part of Covfie is a header-only C++ library. In Covfie, a uniformly
sampled vector field ℝ<sup>n</sup> → ℝ<sup>m</sup> is stored as a scaled and
shifted ℕ<sup>n</sup> → ℝ<sup>m</sup> vector field which allows us to store
data efficiently in GPU memory. Of course, this means that the vector field is
bound to be sampled in a grid: vector fields sampled at arbitrary points are
not supported.

Covfie vector fields support interpolation, using either a nearest neighbour
method, or a linear interpolation method.

Vector fields are used in two different phases. The first phase is the
construction phase, which denotes the process of loading the data about the
vector field into memory in a trivial, backend-independent way. The second
phase is the baking phase, in which the vector field is transferred to one of
the supported back-ends. At this state, the vector field may (or may not)
become read-only. Baked vector fields are accessible on the GPGPU, and should
be trivially passable between device functions.
