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
    std::size_t globalIndex, const cell_container_types::const_view& cells_view,
    vecmem::data::vector_view<unsigned int> channel0,
    vecmem::data::vector_view<unsigned int> channel1,
    vecmem::data::vector_view<unsigned int> cumulsize,
    vecmem::data::vector_view<unsigned int> moduleidx,
    vecmem::data::vector_view<unsigned int> label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    // Initialize the device container for cells
    cell_container_types::const_device cells_device(cells_view);
    vecmem::device_vector<unsigned int> ch0(channel0);
    vecmem::device_vector<unsigned int> ch1(channel1);
    vecmem::device_vector<unsigned int> sum(cumulsize);
    vecmem::device_vector<unsigned int> midx(moduleidx);
    vecmem::device_vector<unsigned int> labels(label_view);

    // Ignore if idx is out of range
    if (globalIndex >= sum.size()-1)
        return;

    // Get the cells for the current module
    const auto& cells = cells_device.at(globalIndex).items;
    printf(" somme module : %u et thread : %d | channel_s 1 : %u | module_id_s : %u | 
     labels : %f \n", sum[globalIndex], globalIndex,  ch1[globalIndex],
     midx[globalIndex],labels[globalIndex] );

    // Vectors used for cluster indices found by sparse CCL
    //vecmem::jagged_device_vector<unsigned int> device_sparse_ccl_indices(
    //    sparse_ccl_indices_view);
    //auto cluster_indices = device_sparse_ccl_indices[globalIndex];

    // Run the sparse CCL algorithm
    unsigned int n_clusters = detail::sparse_ccl(cells, globalIndex, ch0, ch1,
                                        sum, midx, labels);

    // Fill the "number of clusters per
    // module" vector
    vecmem::device_vector<std::size_t> device_clusters_per_module(
        clusters_per_module_view);
    device_clusters_per_module[globalIndex] = n_clusters;
    // printf("module %llu number of clusters %llu \n", globalIndex, device_clusters_per_module[globalIndex]);
}

}  // namespace traccc::device