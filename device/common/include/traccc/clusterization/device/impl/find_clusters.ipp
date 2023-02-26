/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void find_clusters(
    std::size_t globalIndex,
    const CellsView cellsView,
    const ModulesView modulesView,
    vecmem::data::vector_view<unsigned int> clusters_per_module_view) {

    // Initialize the device container for cells
   
    
    vecmem::device_vector<unsigned int> ch0(cellsView.channel0);
    //printf(" hello 3");
    vecmem::device_vector<unsigned int> ch1(cellsView.channel1);
    vecmem::device_vector<unsigned int> sum(modulesView.cells_prefix_sum);
    vecmem::device_vector<unsigned int> midx(cellsView.module_id);
    vecmem::device_vector<unsigned int> labels(cellsView.label);
    //printf(" hello 4");

    // Ignore if idx is out of range
    if (globalIndex >= sum.size())
        return;


    // Run the sparse CCL algorithm
    unsigned int n_clusters = detail::sparse_ccl( globalIndex, ch0, ch1,
                                        sum, midx, labels);

    //if  (globalIndex <20 ) { printf(" midx %llu , channel1 %llu" , ch1[globalIndex], midx[globalIndex] );  } 
    
    vecmem::device_vector<unsigned int> device_clusters_per_module(
        clusters_per_module_view);
    device_clusters_per_module[globalIndex] = n_clusters;
     
}

}  // namespace traccc::device
