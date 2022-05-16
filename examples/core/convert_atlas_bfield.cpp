/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <fstream>
#include <iostream>

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include <covfie/core/backend/builder.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/interpolator/nearest_neighbour.hpp>
#include <covfie/core/backend/vector/input.hpp>
#include <covfie/core/backend/vector/output.hpp>
#include <covfie/core/field.hpp>

void parse_opts(
    int argc, char * argv[], boost::program_options::variables_map & vm
)
{
    boost::program_options::options_description opts("general options");

    opts.add_options()("help", "produce help message")(
        "input,i",
        boost::program_options::value<std::string>()->required(),
        "input magnetic field to read"
    )("output,o",
      boost::program_options::value<std::string>()->required(),
      "output magnetic field to write");

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
        BOOST_LOG_TRIVIAL(error) << e.what();
        std::exit(1);
    }
}

using builder_t = covfie::field<covfie::backend::transformer::affine<
    covfie::backend::transformer::interpolator::nearest_neighbour<
        covfie::backend::builder<
            covfie::backend::vector::input::ulong3,
            covfie::backend::vector::output::float3>>>>;

builder_t read_atlas_bfield(const std::string & fn)
{
    std::ifstream f;

    float minx = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::lowest();
    float miny = std::numeric_limits<float>::max();
    float maxy = std::numeric_limits<float>::lowest();
    float minz = std::numeric_limits<float>::max();
    float maxz = std::numeric_limits<float>::lowest();

    {
        BOOST_LOG_TRIVIAL(info)
            << "Opening magnetic field to compute field limits";

        f.open(fn);

        std::string line;

        BOOST_LOG_TRIVIAL(info) << "Skipping the first four lines (comments)";

        for (std::size_t i = 0; i < 4; ++i) {
            std::getline(f, line);
        }

        float xp, yp, zp;
        float Bx, By, Bz;

        std::size_t n_lines = 0;

        BOOST_LOG_TRIVIAL(info)
            << "Iterating over lines in the magnetic field file";

        /*
         * Read every line, and update our current minima and maxima
         * appropriately.
         */
        while (f >> xp >> yp >> zp >> Bx >> By >> Bz) {
            minx = std::min(minx, xp);
            maxx = std::max(maxx, xp);

            miny = std::min(miny, yp);
            maxy = std::max(maxy, yp);

            minz = std::min(minz, zp);
            maxz = std::max(maxz, zp);

            ++n_lines;
        }

        BOOST_LOG_TRIVIAL(info)
            << "Read " << n_lines << " lines of magnetic field data";

        BOOST_LOG_TRIVIAL(info) << "Closing magnetic field file";

        f.close();
    }

    BOOST_LOG_TRIVIAL(info)
        << "Field dimensions in x = [" << minx << ", " << maxx << "]";
    BOOST_LOG_TRIVIAL(info)
        << "Field dimensions in y = [" << miny << ", " << maxy << "]";
    BOOST_LOG_TRIVIAL(info)
        << "Field dimensions in z = [" << minz << ", " << maxz << "]";

    BOOST_LOG_TRIVIAL(info)
        << "Assuming sample spacing of 100.0 in each dimension";

    /*
     * Now that we have the limits of our field, compute the size in each
     * dimension.
     */
    std::size_t sx = std::lround((maxx - minx) / 100.0) + 1;
    std::size_t sy = std::lround((maxy - miny) / 100.0) + 1;
    std::size_t sz = std::lround((maxz - minz) / 100.0) + 1;

    BOOST_LOG_TRIVIAL(info)
        << "Magnetic field size is " << sx << "x" << sy << "x" << sz;

    BOOST_LOG_TRIVIAL(info) << "Constructing matching field builder...";

    std::array<float, 3> offsets;
    std::array<float, 3> scales;

    scales[0] = (maxx - minx) / (sx - 1);
    offsets[0] = minx;

    scales[1] = (maxy - miny) / (sy - 1);
    offsets[1] = miny;

    scales[2] = (maxz - minz) / (sz - 1);
    offsets[2] = minz;

    builder_t field(
        builder_t::backend_t::configuration_data_t{offsets, scales},
        builder_t::backend_t::backend_t::configuration_data_t{},
        builder_t::backend_t::backend_t::backend_t::configuration_data_t{
            sx, sy, sz}
    );
    builder_t::view_t fv(field);

    {
        BOOST_LOG_TRIVIAL(info) << "Re-opening magnetic field to gather data";

        f.open(fn);

        std::string line;

        BOOST_LOG_TRIVIAL(info) << "Skipping the first four lines (comments)";

        for (std::size_t i = 0; i < 4; ++i) {
            std::getline(f, line);
        }

        float xp, yp, zp;
        float Bx, By, Bz;

        std::size_t n_lines = 0;

        BOOST_LOG_TRIVIAL(info)
            << "Iterating over lines in the magnetic field file";

        /*
         * Read every line, and update our current minima and maxima
         * appropriately.
         */
        while (f >> xp >> yp >> zp >> Bx >> By >> Bz) {
            builder_t::view_t::output_t & p = fv.at(xp, yp, zp);

            p[0] = Bx;
            p[1] = By;
            p[2] = Bz;
        }

        BOOST_LOG_TRIVIAL(info)
            << "Read " << n_lines << " lines of magnetic field data";

        BOOST_LOG_TRIVIAL(info) << "Closing magnetic field file";

        f.close();
    }

    return field;
}

int main(int argc, char ** argv)
{
    boost::program_options::variables_map vm;
    parse_opts(argc, argv, vm);

    BOOST_LOG_TRIVIAL(info) << "Welcome to the covfie magnetic field renderer!";
    BOOST_LOG_TRIVIAL(info) << "Using magnetic field file \""
                            << vm["input"].as<std::string>() << "\"";
    BOOST_LOG_TRIVIAL(info) << "Starting read of input file...";

    builder_t fb = read_atlas_bfield(vm["input"].as<std::string>());

    BOOST_LOG_TRIVIAL(info) << "Writing magnetic field to file \""
                            << vm["output"].as<std::string>() << "\"...";

    std::ofstream fs(vm["output"].as<std::string>(), std::ofstream::binary);

    fb.dump(fs);

    fs.close();

    BOOST_LOG_TRIVIAL(info) << "Rendering complete, goodbye!";

    return 0;
}
