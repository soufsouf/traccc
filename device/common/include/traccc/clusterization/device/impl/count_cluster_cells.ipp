/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>


namespace traccc::device {

TRACCC_DEVICE
inline void count_cluster_cells(
    std::size_t globalIndex,
    vecmem::data::vector_view<unsigned int> cluster_prefix_sum_view,   /// not used 
    const CellsView& cellsView,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view)   {

    // Get the device vector of the cell prefix sum
    vecmem::device_vector<unsigned int> midx(cellsView.module_id);
    vecmem::device_vector<unsigned int> labels(cellsView.label);

    // Ignore if idx is out of range
    if (globalIndex >= labels.size())
        return;

    // Get the indices for the module and the cell in this
    // module, from the prefix sum
   auto module_idx = midx[globalIndex];
   unsigned int cindex = labels[globalIndex] - 1;
    // Vectors used for cluster indices found by sparse CCL

    // Get the cluster prefix sum at this module_idx to know
    // where to write current clusters in the
    // cluster container
    vecmem::device_vector<unsigned int> device_cluster_prefix_sum(
        cluster_prefix_sum_view);

    std::size_t prefix_sum =
        (module_idx == 0 ? 0 : device_cluster_prefix_sum[module_idx - 1]);
    std::size_t cluster_indice = prefix_sum + cindex;  
    // Calculate the number of clusters found for this module from the prefix
    // sums
    std::size_t n_clusters =
        (module_idx == 0 ? device_cluster_prefix_sum[0]
                         : device_cluster_prefix_sum[module_idx] - prefix_sum);

    vecmem::device_vector<unsigned int> device_cluster_sizes(
        cluster_sizes_view);

    // Count the cluster sizes for each position
    if (cindex < n_clusters) {
        //atomicAdd(&device_cluster_sizes[cluster_indice], 1);
        vecmem::device_atomic_ref<unsigned int>(
            device_cluster_sizes[cluster_indice])
            .fetch_add(1);
    }
}  

}  // namespace traccc::device
