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
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/utility/nd_map.hpp>
#include <covfie/core/utility/tuple.hpp>
#include <covfie/core/vector.hpp>
#include <covfie/cuda/error_check.hpp>
#include <covfie/cuda/utility/type_conversion.hpp>

namespace covfie::backend {
enum class cuda_texture_interpolation {
    LINEAR,
    NEAREST_NEIGHBOUR
};

template <
    CONSTRAINT(concepts::vector_descriptor) _input_vector_t,
    CONSTRAINT(concepts::vector_descriptor) _output_vector_t,
    cuda_texture_interpolation _interpolation_method =
        cuda_texture_interpolation::LINEAR>
struct cuda_texture {
    using this_t =
        cuda_texture<_input_vector_t, _output_vector_t, _interpolation_method>;

    static constexpr bool is_initial = true;

    using contravariant_input_t =
        covfie::vector::array_vector_d<_input_vector_t>;
    using covariant_output_t = covfie::vector::array_vector_d<_output_vector_t>;

    using channel_t =
        typename utility::to_cuda_channel_t<_output_vector_t>::type;

    template <typename T>
    using linear_tc = linear<T, float>;

    using configuration_t = std::monostate;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB110001;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;
        owning_data_t(owning_data_t &&) = default;
        owning_data_t & operator=(owning_data_t &&) = default;
        owning_data_t & operator=(const owning_data_t & o) = default;
        owning_data_t(const owning_data_t & o) = default;

        template <
            typename T,
            std::enable_if_t<
                std::is_unsigned_v<
                    typename T::parent_t::contravariant_input_t::scalar_t>,
                bool> = true,
            std::enable_if_t<
                T::parent_t::contravariant_input_t::dimensions ==
                    contravariant_input_t::dimensions,
                bool> = true>
        owning_data_t(const T & o)
        {
            cudaChannelFormatDesc channelDesc =
                cudaCreateChannelDesc<channel_t>();

            typename T::parent_t::non_owning_data_t no(o);

            typename T::parent_t::configuration_t srcSize =
                o.get_configuration();

            cudaExtent extent = make_cudaExtent(
                _input_vector_t::size >= 1 ? srcSize[0] : 0,
                _input_vector_t::size >= 2 ? srcSize[1] : 0,
                _input_vector_t::size >= 3 ? srcSize[2] : 0
            );

            cudaErrorCheck(cudaMalloc3DArray(&m_array, &channelDesc, extent));

            std::unique_ptr<channel_t[]> stage = std::make_unique<channel_t[]>(
                extent.width * extent.height * extent.depth
            );

            using tuple_t = decltype(std::tuple_cat(
                std::declval<
                    std::array<std::size_t, contravariant_input_t::dimensions>>(
                )
            ));
            utility::nd_map(
                std::function<void(tuple_t
                )>([&no, &stage, &srcSize](tuple_t i) -> void {
                    auto ia = utility::to_array(i);
                    typename T::parent_t::covariant_output_t::vector_t v =
                        no.at(ia);

                    typename contravariant_input_t::scalar_t idx = 0;

                    for (std::size_t k = contravariant_input_t::dimensions - 1;
                         k <= contravariant_input_t::dimensions;
                         --k)
                    {
                        typename contravariant_input_t::scalar_t tmp = ia[k];

                        for (std::size_t l = k - 1; l < k; --l) {
                            tmp *= srcSize[l];
                        }

                        idx += tmp;
                    }

                    if constexpr (covariant_output_t::dimensions == 1) {
                        stage[idx] = v[0];
                    } else if constexpr (covariant_output_t::dimensions == 2) {
                        stage[idx].x = v[0];
                        stage[idx].y = v[1];
                    } else if constexpr (covariant_output_t::dimensions == 3) {
                        stage[idx].x = v[0];
                        stage[idx].y = v[1];
                        stage[idx].z = v[2];
                        stage[idx].w = 0.f;
                    } else if constexpr (covariant_output_t::dimensions == 4) {
                        stage[idx].x = v[0];
                        stage[idx].y = v[1];
                        stage[idx].z = v[2];
                        stage[idx].w = v[3];
                    }
                }),
                std::tuple_cat(srcSize)
            );

            if constexpr (_input_vector_t::size == 2) {
                cudaErrorCheck(cudaMemcpy2DToArray(
                    m_array,
                    0,
                    0,
                    stage.get(),
                    extent.width,
                    extent.width,
                    extent.height,
                    cudaMemcpyHostToDevice
                ));
            } else if constexpr (_input_vector_t::size == 3) {
                cudaMemcpy3DParms copyParams;
                // cudaMemcpy3DParms copyParams = {0};
                memset(&copyParams, 0, sizeof(cudaMemcpy3DParms));
                copyParams.srcPtr = make_cudaPitchedPtr(
                    stage.get(),
                    extent.width * sizeof(channel_t),
                    extent.width,
                    extent.height
                );
                copyParams.dstArray = m_array;
                copyParams.extent = extent;
                copyParams.kind = cudaMemcpyHostToDevice;
                cudaErrorCheck(cudaMemcpy3D(&copyParams));
            }

            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = m_array;

            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));

            // TODO: Make configurable
            for (std::size_t i = 0; i < _input_vector_t::size; ++i) {
                texDesc.addressMode[i] = cudaAddressModeClamp;
            }

            // TODO: Make configurable
            if (_interpolation_method == cuda_texture_interpolation::LINEAR) {
                texDesc.filterMode = cudaFilterModeLinear;
            } else if (_interpolation_method == cuda_texture_interpolation::NEAREST_NEIGHBOUR)
            {
                texDesc.filterMode = cudaFilterModePoint;
            }
            texDesc.readMode = cudaReadModeElementType;

            cudaErrorCheck(
                cudaCreateTextureObject(&m_tex, &resDesc, &texDesc, nullptr)
            );
        }

        template <typename T>
        owning_data_t(parameter_pack<T> && i)
            : owning_data_t(std::move(i.x))
        {
        }

        ~owning_data_t()
        {
            cudaErrorCheck(cudaDestroyTextureObject(m_tex));
            cudaErrorCheck(cudaFreeArray(m_array));
        }

        configuration_t get_configuration() const
        {
            return {};
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            throw std::invalid_argument("Cannot perform IO on texture memory.");

            return owning_data_t();
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            throw std::invalid_argument("Cannot perform IO on texture memory.");
        }

        cudaArray_t m_array;
        cudaTextureObject_t m_tex;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & o)
            : m_tex(o.m_tex)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t i) const
        {
            channel_t r;

            if constexpr (_input_vector_t::size == 1) {
                r = tex1D<channel_t>(m_tex, i[0] + 0.5);
            } else if constexpr (_input_vector_t::size == 2) {
                r = tex2D<channel_t>(m_tex, i[0] + 0.5, i[1] + 0.5);
            } else if constexpr (_input_vector_t::size == 3) {
                r = tex3D<channel_t>(m_tex, i[0] + 0.5, i[1] + 0.5, i[2] + 0.5);
            }

            if constexpr (_output_vector_t::size == 1) {
                return {r.x};
            } else if constexpr (_output_vector_t::size == 2) {
                return {r.x, r.y};
            } else if constexpr (_output_vector_t::size == 3) {
                return {r.x, r.y, r.z};
            } else if constexpr (_output_vector_t::size == 4) {
                return {r.x, r.y, r.z, r.w};
            }

            return {};
        }

        cudaTextureObject_t m_tex;
    };
};
}
