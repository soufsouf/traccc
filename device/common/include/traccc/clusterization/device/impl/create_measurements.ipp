/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cassert>

namespace traccc::device {

TRACCC_DEVICE
inline void create_measurements(
    std::size_t globalIndex, 
    vecmem::data::vector_view<unsigned int > moduleidx,
    vecmem::data::vector_view<scalar> activation_cell,
    vecmem::data::vector_view<unsigned int> channel0,
    vecmem::data::vector_view<unsigned int> channel1,
    vecmem::data::vector_view<unsigned int > clusters_view,
    vecmem::data::vector_view<unsigned int > cel_cl_ps, // cell_cluster_prefix_sum
    const cell_container_types::const_view& cells_view,
    measurement_container_types::view measurements_view) {

    // Initialize device vector that gives us the execution range
    vecmem::device_vector<unsigned int> midx(moduleidx);
    vecmem::device_vector<scalar> activation(activation_cell);
    vecmem::device_vector<unsigned int> ch0(channel0);
    vecmem::device_vector<unsigned int> ch1(channel1);
 vecmem::device_vector<unsigned int> clusters_device(clusters_view);
    vecmem::device_vector<unsigned int> cells_per_cluster_prefix_sum(cel_cl_ps);
    cell_container_types::const_device cells_device(cells_view);
    measurement_container_types::device measurements_device(measurements_view);
    
    
    // Ignore if idx is out of range
    if (globalIndex >= clusters_device.size())
        return;

    // Create other device containers
    

    // items: cluster of cells at current idx
    // header: module idx
    //obtenir les cells de cluster: remplacer par deux vec: 
    //on met dans le premier l'indice de debut des cells d'un cluster dans le vecteur device_clusters 
    //et dans le deuxieme prefix sum on peut obtenir le nombre de cells par cluster 
    std::size_t idx_cluster = (globalIndex == 0 ? 0 : cells_per_cluster_prefix_sum[globalIndex - 1]  );; // l'indice debut cluster dans le vecteur device_cluster
    unsigned int idx_cell = clusters_device[idx_cluster];
    std::size_t module_link = midx[idx_cell];
    std::size_t nbr_cell_per_cluster = cells_per_cluster_prefix_sum[globalIndex]- idx_cluster;
    const auto& module = cells_device.at(module_link).header; // c quoi header

    // Should not happen
    assert(clusters_device.empty() == false);
   
    // Fill measurement from cluster
    detail::fill_measurement(measurements_device, clusters_device, idx_cluster, 
     nbr_cell_per_cluster, activation,ch0 , ch1 , module, module_link,globalIndex);
}

}  // namespace traccc::device
