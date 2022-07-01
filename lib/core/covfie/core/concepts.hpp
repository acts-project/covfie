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

#include <fstream>

#include <covfie/core/definitions.hpp>

#if __cpp_concepts >= 201907L
#include <concepts>
namespace covfie::concepts {
template <typename T>
concept field_backend = requires
{
    /*
     * Check whether the backend has the required type definitions, which are
     * the input and output types of the contravariant and covariant parts of
     * the layer.
     */
    typename T::contravariant_input_t;
    typename T::covariant_output_t;

    /*
     * Confirm that the backend has both an owning and a non-owning data type.
     */
    typename T::owning_data_t;
    typename T::non_owning_data_t;

    /*
     * The non-owning data which we use on the GPU must be extremely simple,
     * because we cannot model destructors and such there properly.
     */
    requires std::is_trivially_destructible_v<typename T::non_owning_data_t>;
    requires
        std::is_trivially_copy_constructible_v<typename T::non_owning_data_t>;
    requires
        std::is_trivially_move_constructible_v<typename T::non_owning_data_t>;

    /*
     * Check whether the owning data type can be read from a file.
     */
    requires requires(std::ifstream & fs)
    {
        {typename T::owning_data_t(fs)};
    };

    requires requires(const typename T::owning_data_t & d)
    {
        /*
         * Check whether a non-owning data type can be constructed from the
         * owning variant.
         */
        {typename T::non_owning_data_t(d)};

        /*
         * Check whether an owning data type can be written to disk.
         */
        requires requires(std::ofstream & fs)
        {
            {d.dump(fs)};
        };
    };

    /*
     * Check whether a non-owning object allows us to look up the magnetic
     * field, and whether that operation gives the correct result.
     */
    requires requires(const typename T::non_owning_data_t & d)
    {
        {
            d.at(std::declval<typename T::contravariant_input_t::vector_t>())
            } -> std::same_as<typename T::covariant_output_t::vector_t>;
    };

    /*
     * Make sure that owning data types are copyable so we can easily copy
     * fields.
     */
    requires std::copyable<typename T::owning_data_t>;
};

template <typename T>
concept vector_descriptor = true;
}
#endif
