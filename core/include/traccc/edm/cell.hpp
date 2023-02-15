/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/geometry/pixel_data.hpp"

// System include(s).
#include <cmath>
#include <limits>

namespace traccc {

/// Definition for one detector cell
///
/// It comes with two integer channel identifiers, an "activation value"
/// and a time stamp.
///
struct cell {
    channel_id channel0 = 0;
    channel_id channel1 = 0;
    scalar activation = 0.;
    scalar time = 0.;
};

/// Comparison / ordering operator for cells
TRACCC_HOST_DEVICE
inline bool operator<(const cell& lhs, const cell& rhs) {

    if (lhs.channel0 != rhs.channel0) {
        return (lhs.channel0 < rhs.channel0);
    } else if (lhs.channel1 != rhs.channel1) {
        return (lhs.channel1 < rhs.channel1);
    } else {
        return lhs.activation < rhs.activation;
    }
}

/// Equality operator for cells
TRACCC_HOST_DEVICE
inline bool operator==(const cell& lhs, const cell& rhs) {

    return (
        (lhs.channel0 == rhs.channel0) && (lhs.channel1 == rhs.channel1) &&
        (std::abs(lhs.activation - rhs.activation) < traccc::float_epsilon) &&
        (std::abs(lhs.time - rhs.time) < traccc::float_epsilon));
}

/// Header information for all of the cells in a specific detector module
///
/// It is handled separately from the list of all of the cells belonging to
/// the detector module, to be able to lay out the data in memory in a way
/// that is more friendly towards accelerators.
///
using scalar = TRACCC_CUSTOM_SCALARTYPE;
struct cell_module {

    geometry_id module = 0;
    transform3 placement = transform3{};
    scalar threshold = 0;

    pixel_data pixel;

};  // struct cell_module
using scalar_vec = std::vector<scalar>;
using geometry_id_vec = std::vector<geometry_id>;
using transform3_vec = std::vector<transform3>;
using pixel_data_vec = std::vector<pixel_data>;

struct headerVec{
geometry_id_vec module;
transform3_vec placement;
scalar_vec threshold;
pixel_data_vec pixel;
};
using scalar_buf = vecmem::data::vector_buffer<scalar>;
using geometry_id_buf = vecmem::data::vector_buffer<geometry_id>;
using transform3_buf = vecmem::data::vector_buffer<transform3>;
using pixel_data_buf = vecmem::data::vector_buffer<pixel_data>;
struct headerBuff{
geometry_id_buf module;
transform3_buf placement;
scalar_buf threshold;
pixel_data_buf pixel;
};
using scalar_view = vecmem::data::vector_view<scalar>;
using geometry_id_view = vecmem::data::vector_view<geometry_id>;
using transform3_view = vecmem::data::vector_view<transform3>;
using pixel_data_view = vecmem::data::vector_view<pixel_data>;  

struct headerView{
 geometry_id_view module;
transform3_view placement;
scalar_view threshold;
pixel_data_view pixel;   
};
using scalar_device = vecmem::device_vector<scalar>;
using geometry_id_device = vecmem::device_vector<geometry_id>;
using transform3_device = vecmem::device_vector<transform3>;
using pixel_data_device = vecmem::device_vector<pixel_data>;
struct headerVecDevice{
geometry_id_device module;
transform3_device placement;
scalar_device threshold;
pixel_data_device pixel; 
};


using int_vec = std::vector<unsigned int>;

struct CellVec {
    int_vec channel0;
    int_vec channel1;
    scalar_vec activation;
    scalar_vec time;
    int_vec module_id;
    int_vec cluster_id;
    std::size_t size;
    std::size_t module_size;
};

using int_buf = vecmem::data::vector_buffer<unsigned int>;

struct CellBuf {
    int_buf channel0;
    int_buf channel1;
    scalar_buf activation;
    scalar_buf time;
    int_buf module_id;
    int_buf cluster_id;
};

using int_view = vecmem::data::vector_view<unsigned int>;

struct CellView {
    int_view channel0;
    int_view channel1;
    scalar_view activation;
    scalar_view time;
    int_view module_id;
    int_view cluster_id;
};

using int_device = vecmem::device_vector<unsigned int>;

struct CellVecDevice {
    int_device channel0;
    int_device channel1;
    scalar_device activation;
    scalar_device time;
    int_device module_id;
    int_device cluster_id;
};

struct Cluster {
    int_view cells_prefix_sum;
    int_view module_id;
};

struct ModuleView {
    int_view cells_prefix_sum;
    int_view clusters_prefix_sum;
};

struct ModuleVec {
    int_vec cells_prefix_sum;
    int_vec clusters_prefix_sum;
    std::size_t size;
};
struct ModuleBuf {
    int_buf cells_prefix_sum;
    int_buf clusters_prefix_sum;
};
struct ModuleVecDevice {
    int_device cells_prefix_sum;
    int_device clusters_prefix_sum;
};

/// Equality operator for cell module
TRACCC_HOST_DEVICE
inline bool operator==(const cell_module& lhs, const cell_module& rhs) {
    return lhs.module == rhs.module;
}

/// Declare all cell collection types
using cell_collection_types = collection_types<cell>;
/// Declare all cell container types
using cell_container_types = container_types<cell_module, cell>;

}  // namespace traccc
