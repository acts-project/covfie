/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory>

#include <cuda_runtime.h>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/vector.hpp>
#include <covfie/cuda/error_check.hpp>
#include <covfie/cuda/utility/memory.hpp>
#include <covfie/cuda/utility/unique_ptr.hpp>

namespace covfie::backend {
template <
    CONSTRAINT(concepts::vector_descriptor) _output_vector_t,
    typename _index_t = std::size_t>
struct cuda_device_array {
    using this_t = cuda_device_array<_output_vector_t, _index_t>;

    static constexpr bool is_initial = true;

    using contravariant_input_t =
        covfie::vector::scalar_d<covfie::vector::vector_d<_index_t, 1>>;
    using covariant_output_t =
        covfie::vector::array_reference_vector_d<_output_vector_t>;

    using output_vector_t = _output_vector_t;

    using value_t = typename output_vector_t::type[output_vector_t::size];
    using vector_t = std::decay_t<typename covariant_output_t::vector_t>;

    using configuration_t = utility::nd_size<1>;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB110000;

    struct owning_data_t {
        using parent_t = this_t;

        explicit owning_data_t(parameter_pack<configuration_t> && args)
            : m_size(args.x[0])
            , m_ptr(utility::cuda::device_allocate<vector_t[]>(m_size))
        {
        }

        explicit owning_data_t(
            std::size_t size, std::unique_ptr<vector_t[]> && ptr
        )
            : m_size(size)
            , m_ptr(utility::cuda::device_copy(std::move(ptr), size))
        {
        }

        configuration_t get_configuration() const
        {
            return {m_size};
        }

        std::size_t m_size;
        utility::cuda::unique_device_ptr<vector_t[]> m_ptr;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr.get())
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t i) const
        {
            return m_ptr[i];
        }

        vector_t * m_ptr;
    };
};
}
