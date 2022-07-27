#include <covfie/core/algebra/affine.hpp>
#include <covfie/core/backend/initial/array.hpp>
#include <covfie/core/backend/initial/constant.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/interpolator/nearest_neighbour.hpp>
#include <covfie/core/backend/transformer/layout/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>

using backend_t1 = covfie::backend::transformer::affine<
    covfie::backend::transformer::interpolator::nearest_neighbour<
        covfie::backend::layout::strided<
            covfie::vector::ulong3,
            covfie::backend::storage::array<covfie::vector::float3>>>>;

using backend_t2 =
    covfie::backend::transformer::interpolator::nearest_neighbour<
        covfie::backend::layout::strided<
            covfie::vector::ulong3,
            covfie::backend::storage::array<covfie::vector::float3>>>;

using backend_t3 = covfie::backend::layout::strided<
    covfie::vector::ulong3,
    covfie::backend::storage::array<covfie::vector::float3>>;

using backend_t4 = covfie::backend::storage::array<covfie::vector::float3>;

using approx_backend_t = covfie::backend::transformer::affine<
    covfie::backend::transformer::interpolator::nearest_neighbour<
        covfie::backend::layout::strided<
            covfie::vector::ulong3,
            covfie::backend::
                constant<covfie::vector::ulong1, covfie::vector::float3>>>>;

template covfie::field_view<approx_backend_t>::output_t
covfie::field_view<approx_backend_t>::at<float, float, float>(
    float, float, float
) const;
template covfie::field_view<backend_t1>::output_t
covfie::field_view<backend_t1>::at<float, float, float>(float, float, float)
    const;
template covfie::field_view<backend_t2>::output_t
covfie::field_view<backend_t2>::at<float, float, float>(float, float, float)
    const;
template covfie::field_view<backend_t3>::output_t
    covfie::field_view<backend_t3>::at<ulong>(ulong, ulong, ulong) const;
template covfie::field_view<backend_t4>::output_t
    covfie::field_view<backend_t4>::at<ulong>(ulong) const;
