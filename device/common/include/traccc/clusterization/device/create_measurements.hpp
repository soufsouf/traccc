/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/detail/measurement_creation_helper.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function used for creating the 2D measurement objects out of the clusters in
/// each module
///
/// The output is a measurement container with 1 measurement corresponding to 1
/// cluster
///
/// @param[in] globalIndex          The index for the current thread
/// @param[in] clusters_view        Container storing the cells for every
/// cluster
/// @param[in] cells_view           The cells for each module
/// @param[out] measurements_view   Container storing the created measurements
/// for each module
///
TRACCC_HOST_DEVICE
inline void create_measurements(
    std::size_t globalIndex, 
    vecmem::data::vector_view<unsigned int > moduleidx,
    vecmem::data::vector_view<scalar> activation_cell,
    vecmem::data::vector_view<unsigned int> channel0,
    vecmem::data::vector_view<unsigned int> channel1,
    vecmem::data::vector_view<unsigned int > clusters_view,
    vecmem::data::vector_view<unsigned int > cel_cl_ps, // cell_cluster_prefix_sum
    vecmem::data::vector_view<unsigned int > emplacement, //nouveau tableau de taille n_clusters : chaque case contient l'indice de debut de cluster
    const cell_container_types::const_view& cells_view,
    measurement_container_types::view measurements_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/create_measurements.ipp"
