/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/detail/sparse_ccl.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function that finds the clusters using sparse_ccl algorithm
///
/// It saves the cluster indices for each module in a jagged vector
/// and it counts how many clusters in total were found
///
/// @param[in] globalIndex                  The index of the current thread
/// @param[in] cells_view                   The cells for each module

/// corresponding clusters
/// @param[in] channel0
/// @param[in] channel1
/// @param[in] cumulsize
/// @param[in] moduleidx
/// @param[out] label_view

/// @param[out] clusters_per_module_view    Vector of numbers of clusters found
/// in each module
///
TRACCC_HOST_DEVICE
inline void find_clusters(
    std::size_t globalIndex, 
    const CellsView cellsView,
    const ModulesView modulesView,
    vecmem::data::vector_view<unsigned int> clusters_per_module_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/find_clusters.ipp"