/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

    struct cell_struct {
    unsigned int channel0 = 0;
    unsigned int channel1 = 0;
    scalar activation = 0.;
    scalar time = 0.;
};

/// Function used for the filling the cluster container with corresponding cells
///
/// The output is the cluster container with module indices as headers and
/// clusters of cells as items Since the headers are module_idx, and not
/// cluster_idx, there can be multiple same module_idx next to each other
///
/// @param[in] globalIndex              The index for the current thread
/// @param[in] moduleidx               The cells for each module
/// @param[in] label_view     Jagged vector that maps cells to
/// corresponding clusters
/// @param[in] cluster_prefix_sum_view  Prefix sum vector made out of number of
/// clusters in each module
/// @param[in] cluster_idx_atomic    Prefix sum for iterating over all the
/// cells
/// @param[in] cells_cl_prefix_sum
/// @param[out] clusters_view           Container storing the cells for every
/// cluster
///
TRACCC_HOST
inline void connect_components(
    std::size_t globalIndex, vecmem::data::vector_view<unsigned int> moduleidx,
     vecmem::data::vector_view<unsigned int> label_view,
     vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
     vecmem::data::vector_view<unsigned int> cluster_idx_atomic,
     vecmem::data::vector_view<unsigned int> cells_cl_prefix_sum,
    vecmem::data::vector_view<unsigned int> clusters_view);



TRACCC_DEVICE
inline void connect_components(
    std::size_t globalIndex, 
    vecmem::data::vector_view<unsigned int> channel0,
    vecmem::data::vector_view<unsigned int> channel1,
    vecmem::data::vector_view<TRACCC_CUSTOM_SCALARTYPE> activation_cell, 
    vecmem::data::vector_view<unsigned int > moduleidx,
    vecmem::data::vector_view<unsigned int > celllabel,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,//cluster per module
    vecmem::data::vector_view<unsigned int > cluster_atomic,
    vecmem::data::jagged_vector_view<cell_struct>& clusters_view, int eee);

// Include the implementation.
#include "traccc/clusterization/device/impl/connect_components.ipp"

}  // namespace traccc::device