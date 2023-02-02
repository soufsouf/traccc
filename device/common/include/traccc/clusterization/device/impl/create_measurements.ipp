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
using scalar = TRACCC_CUSTOM_SCALARTYPE;
TRACCC_DEVICE
inline void create_measurements(
    std::size_t globalIndex, 
    vecmem::data::vector_view<unsigned int > moduleidx,
    vecmem::data::vector_view<scalar> activation_cell,
    vecmem::data::vector_view<unsigned int> channel0,
    vecmem::data::vector_view<unsigned int> channel1,
    vecmem::data::vector_view<unsigned int > clusters_view,
    vecmem::data::vector_view<unsigned int > cel_cl_ps, // cell_cluster_prefix_sum
    const cell_container_types::const_view cells_view,
     vecmem::data::vector_view<unsigned int >& Clusters_module_link,
     vecmem::data::vector_view<point2 > &measurement_local,
      vecmem::data::vector_view<point2 >& measurement_variance) {

    // Initialize device vector that gives us the execution range
    vecmem::device_vector<unsigned int> midx(moduleidx);
    vecmem::device_vector<scalar> activation(activation_cell);
    vecmem::device_vector<unsigned int> ch0(channel0);
    vecmem::device_vector<unsigned int> ch1(channel1);
    vecmem::device_vector<unsigned int> clusters_device(clusters_view);
    vecmem::device_vector<unsigned int> cells_per_cluster_prefix_sum(cel_cl_ps);
    cell_container_types::const_device cells_device(cells_view);
    vecmem::device_vector<unsigned int> Cl_module_link(Clusters_module_link);
    vecmem::device_vector<point2> local_measurement(measurement_local);
    vecmem::device_vector<point2> variance_measurement(measurement_variance);
    
    
    // Ignore if idx is out of range
    if (globalIndex >= cells_per_cluster_prefix_sum.size()) /// faux 
        return;

    // Create other device containers
    

    // items: cluster of cells at current idx
    // header: module idx
    //obtenir les cells de cluster: remplacer par deux vec: 
    //on met dans le premier l'indice de debut des cells d'un cluster dans le vecteur device_clusters 
    //et dans le deuxieme prefix sum on peut obtenir le nombre de cells par cluster 
    std::size_t idx_cluster = (globalIndex == 0 ? 0 : cells_per_cluster_prefix_sum[globalIndex - 1]  ); // l'indice debut cluster dans le vecteur device_cluster
    unsigned int idx_cell = clusters_device[idx_cluster];   //esq idx_cluster = idx_cell (checkout)
    std::size_t module_link = midx[idx_cell];
    Cl_module_link[globalIndex] = module_link;
    std::size_t nbr_cell_per_cluster = cells_per_cluster_prefix_sum[globalIndex] - idx_cluster;
    auto &module = cells_device.at(module_link).header; // c quoi header

    /*printf("th %llu cluster %llu cell %u module %llu nbr cell per cluster %llu\n",
            globalIndex, idx_cluster, idx_cell, module_link, nbr_cell_per_cluster);*/

    // Should not happen
    assert(clusters_device.empty() == false);
   
    // Fill measurement from cluster
    

    detail::fill_measurement(local_measurement,variance_measurement, clusters_device, idx_cluster, 
        nbr_cell_per_cluster, activation,ch0 , ch1 , module, module_link, globalIndex);
}

}  // namespace traccc::device