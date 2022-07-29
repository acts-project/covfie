/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cstddef>
#include <fstream>
#include <iostream>

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/interpolator/linear.hpp>
#include <covfie/core/backend/transformer/layout/strided.hpp>
#include <covfie/core/backend/transformer/ownership/reference.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/utility/nd_size.hpp>

void parse_opts(
    int argc, char * argv[], boost::program_options::variables_map & vm
)
{
    boost::program_options::options_description opts("general options");

    opts.add_options()("help", "produce help message")(
        "input,i",
        boost::program_options::value<std::string>()->required(),
        "input vector field to read"
    )("output,o",
      boost::program_options::value<std::string>()->required(),
      "output vector field to write"
    )("axis,a",
      boost::program_options::value<std::string>()->required(),
      "axis along which to slice (\"x\", \"y\", or \"z\")"
    )("slice,s",
      boost::program_options::value<unsigned long>()->required(),
      "slice coordinate along the specified axis");

    boost::program_options::parsed_options parsed =
        boost::program_options::command_line_parser(argc, argv)
            .options(opts)
            .run();

    boost::program_options::store(parsed, vm);

    if (vm.count("help")) {
        std::cout << opts << std::endl;
        std::exit(0);
    }

    try {
        boost::program_options::notify(vm);
    } catch (boost::program_options::required_option & e) {
        BOOST_LOG_TRIVIAL(fatal) << e.what();
        std::exit(1);
    }

    if (vm["axis"].as<std::string>() != "x" &&
        vm["axis"].as<std::string>() != "y" &&
        vm["axis"].as<std::string>() != "z")
    {
        BOOST_LOG_TRIVIAL(fatal) << "Axis specification must be x, y, or z!";
        std::exit(1);
    }
}

int main(int argc, char ** argv)
{
    using core_t = covfie::backend::layout::strided<
        covfie::vector::ulong3,
        covfie::backend::storage::array<covfie::vector::float3>>;
    using field_t1 = covfie::field<covfie::backend::transformer::affine<
        covfie::backend::transformer::interpolator::linear<core_t>>>;
    using field_t2 = covfie::field<
        covfie::backend::transformer::ownership::reference<core_t>>;
    using field_t3 = covfie::field<covfie::backend::layout::strided<
        covfie::vector::ulong2,
        covfie::backend::storage::array<covfie::vector::float3>>>;

    boost::program_options::variables_map vm;
    parse_opts(argc, argv, vm);

    BOOST_LOG_TRIVIAL(info) << "Welcome to the covfie vector field slicer!";
    BOOST_LOG_TRIVIAL(info) << "Using vector field file \""
                            << vm["input"].as<std::string>() << "\"";
    BOOST_LOG_TRIVIAL(info) << "Starting read of input file...";

    std::ifstream ifs(vm["input"].as<std::string>(), std::ifstream::binary);

    if (!ifs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open input file "
                                 << vm["input"].as<std::string>() << "!";
        std::exit(1);
    }

    field_t1 f(ifs);
    ifs.close();

    BOOST_LOG_TRIVIAL(info) << "Fetching integral coordinate part of field...";

    const core_t::owning_data_t & core_data =
        f.backend().get_backend().get_backend();

    field_t2 nf(core_data);
    field_t2::view_t ifv(nf);

    BOOST_LOG_TRIVIAL(info) << "Building new output vector field...";

    covfie::utility::nd_size<3> in_size =
        nf.backend().get_backend().get_configuration();
    covfie::utility::nd_size<2> out_size;

    if (vm["axis"].as<std::string>() == "x") {
        out_size = {in_size[1], in_size[2]};
    } else if (vm["axis"].as<std::string>() == "y") {
        out_size = {in_size[0], in_size[2]};
    } else if (vm["axis"].as<std::string>() == "z") {
        out_size = {in_size[0], in_size[1]};
    }

    field_t3 of(field_t3::backend_t::configuration_t{out_size});

    BOOST_LOG_TRIVIAL(info) << "Creating vector field views...";

    field_t3::view_t ofv(of);

    BOOST_LOG_TRIVIAL(info) << "Slicing vector field...";

    if (vm["axis"].as<std::string>() == "x") {
        for (unsigned long x = 0; x < out_size[0]; ++x) {
            for (unsigned long y = 0; y < out_size[1]; ++y) {
                ofv.at(x, y) = ifv.at(vm["slice"].as<unsigned long>(), x, y);
            }
        }
    } else if (vm["axis"].as<std::string>() == "y") {
        for (unsigned long x = 0; x < out_size[0]; ++x) {
            for (unsigned long y = 0; y < out_size[1]; ++y) {
                ofv.at(x, y) = ifv.at(x, vm["slice"].as<unsigned long>(), y);
            }
        }
    } else if (vm["axis"].as<std::string>() == "z") {
        for (unsigned long x = 0; x < out_size[0]; ++x) {
            for (unsigned long y = 0; y < out_size[1]; ++y) {
                ofv.at(x, y) = ifv.at(x, y, vm["slice"].as<unsigned long>());
            }
        }
    }

    BOOST_LOG_TRIVIAL(info) << "Saving result to file \""
                            << vm["output"].as<std::string>() << "\"...";

    std::ofstream ofs(vm["output"].as<std::string>(), std::ifstream::binary);
    of.dump(ofs);
    ofs.close();

    BOOST_LOG_TRIVIAL(info) << "Procedure complete, goodbye!";
}
