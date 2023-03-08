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
using index_t = unsigned short;
struct idx_cluster {
    unsigned int id_cluster = 0 ;
    unsigned int write = 0 ;
    unsigned int module_link ;
    unsigned int emplacement ;
};
struct grp_cluster {
    unsigned int write = 0 ;
    index_t cluster_cell ;
    unsigned int nbr_cell = 0;
    
};
struct writeEqual {
  unsigned int write;
  __device__ writeEqual(unsigned int a) : write(a) {}
  __device__ bool operator()(const grp_cluster& p) const {
    return p.write == write;
  }
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
struct cell_module {

    geometry_id module = 0;
    transform3 placement = transform3{};
    scalar threshold = 0;

    pixel_data pixel;

};  // struct cell_module

/// Declare all cell module collection types
using cell_module_collection_types = collection_types<cell_module>;

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
