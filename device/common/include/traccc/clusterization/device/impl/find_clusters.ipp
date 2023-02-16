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
    const CellView& cellView,
    const ModuleView& moduleView,
    vecmem::data::vector_view<unsigned int> label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    // Initialize the device container for cells
   
    
    vecmem::device_vector<unsigned int> ch0(cellView.channel0);
    //printf(" hello 3");
    vecmem::device_vector<unsigned int> ch1(cellView.channel1);
    vecmem::device_vector<unsigned int> sum(moduleView.cells_prefix_sum);
    vecmem::device_vector<unsigned int> midx(cellView.module_id);
    vecmem::device_vector<unsigned int> labels(label_view);
    //printf(" hello 4");

    // Ignore if idx is out of range
    if (globalIndex >= sum.size())
        return;


    // Run the sparse CCL algorithm
    unsigned int n_clusters = detail::sparse_ccl( globalIndex, ch0, ch1,
                                        sum, midx, labels);

    if  (globalIndex <= 5 ) { printf(" channel1 %llu" , ch1[globalIndex] ); 
                              printf(" globalIndex %llu" , globalIndex ); }
   
    vecmem::device_vector<std::size_t> device_clusters_per_module(
        clusters_per_module_view);
    device_clusters_per_module[globalIndex] = n_clusters;
     
}

}  // namespace traccc::device