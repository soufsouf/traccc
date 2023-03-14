/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/alt_cell.hpp"
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
// System include(s).
#include <cstddef>
#include <cuda_runtime_api.h>
namespace traccc::device {

/// Function which looks for cells which share the same "parent" index and
/// aggregates them into a cluster.
///
/// @param[in] cells    collection of cells
/// @param[in] modules  collection of modules to which the cells are linked to
/// @param[in] f        array of "parent" indices for all cells in this
/// partition
/// @param[in] start    partition start point this cell belongs to
/// @param[in] end      partition end point this cell belongs to
/// @param[in] cid      current cell id
/// @param[out] out     cluster to fill
TRACCC_HOST_DEVICE
inline void aggregate_cluster(
    const texture<traccc::alt_cell, 1, cudaReadModeElementType> cells_device,
    const cell_module_collection_types::const_device& modules,
     unsigned short* f,
    const unsigned int start, const unsigned int end, const unsigned short cid,
    spacepoint_collection_types::device spacepoints_device,
     vecmem::data::vector_view<unsigned int> cell_links,
    const unsigned int link);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/aggregate_cluster.ipp"