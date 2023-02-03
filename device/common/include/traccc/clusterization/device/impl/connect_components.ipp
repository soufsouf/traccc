/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST 
inline void connect_components(
    std::size_t globalIndex, const cell_container_types::const_view& cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    cluster_container_types::view clusters_view) {

    // Get device vector of the cells prefix sum
    vecmem::device_vector<const device::prefix_sum_element_t> cells_prefix_sum(
        cells_prefix_sum_view);

    if (globalIndex >= cells_prefix_sum.size())
        return;

    // Get the indices for the module idx and the cell idx
    auto module_idx = cells_prefix_sum[globalIndex].first;
    auto cell_idx = cells_prefix_sum[globalIndex].second;

    // Initialize the device containers for cells and clusters
    cell_container_types::const_device cells_device(cells_view);
    cluster_container_types::device clusters_device(clusters_view);

    // Get the cells for the current module idx
    const auto& cells = cells_device.at(module_idx).items;

    // Vectors used for cluster indices found by sparse CCL
    vecmem::jagged_device_vector<unsigned int> device_sparse_ccl_indices(
        sparse_ccl_indices_view);
    const auto& cluster_indices = device_sparse_ccl_indices.at(module_idx);

    // Get the cluster prefix sum for this module idx
    vecmem::device_vector<std::size_t> device_cluster_prefix_sum(
        cluster_prefix_sum_view);
    const std::size_t prefix_sum =
        (module_idx == 0 ? 0 : device_cluster_prefix_sum[module_idx - 1]);

    // Calculate the number of clusters found for this module from the prefix
    // sums
    const unsigned int n_clusters =
        (module_idx == 0 ? device_cluster_prefix_sum[module_idx]
                         : device_cluster_prefix_sum[module_idx] -
                               device_cluster_prefix_sum[module_idx - 1]);

    // Push back the cells to the correct item vector indicated
    // by the cluster prefix sum
    unsigned int cindex = cluster_indices[cell_idx] - 1;
    if (cindex < n_clusters) {
        clusters_device[prefix_sum + cindex].header = module_idx;
        clusters_device[prefix_sum + cindex].items.push_back(cells[cell_idx]);  
    }
}

TRACCC_DEVICE
inline void connect_components(
    std::size_t globalIndex,
    vecmem::data::vector_view<unsigned int> channel0,
    vecmem::data::vector_view<unsigned int> channel1,
    vecmem::data::vector_view<scalar> activation_cell, 
    vecmem::data::vector_view<unsigned int > moduleidx,
    vecmem::data::vector_view<unsigned int > celllabel,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,//cluster per module
    vecmem::data::vector_view<unsigned int > cluster_atomic,
    vecmem::data::jagged_vector_view<unsigned int> clusters_view) {

    // Get device vector of the cells prefix sum
    vecmem::device_vector<scalar> activation(activation_cell);
    vecmem::device_vector<unsigned int> ch0(channel0);
    vecmem::device_vector<unsigned int> ch1(channel1);
    vecmem::device_vector<unsigned int> midx(moduleidx);
    vecmem::jagged_device_vector<unsigned int> clusters_device(clusters_view);
    vecmem::device_vector<unsigned int> labels(celllabel);
    vecmem::device_vector<unsigned int> cluster_index_atomic(cluster_atomic);
    vecmem::device_vector<std::size_t> device_cluster_prefix_sum(cluster_prefix_sum_view);
    
   

    if (globalIndex >= labels.size())
        return;

    // Get the indices for the module idx and the cell idx
    auto module_idx = midx[globalIndex];

    // Get the cells for the current module idx
    

    // Vectors used for cluster indices found by sparse CCL
    unsigned int cindex = labels[globalIndex] - 1;

    // Get the cluster prefix sum for this module idx
    
    const std::size_t prefix_sum = (module_idx == 0 ? 0 : module_idx - 1);
    auto cluster_indice = device_cluster_prefix_sum[prefix_sum]+ cindex;

    // Calculate the number of clusters found for this module from the prefix
    // sums
    const unsigned int n_clusters =
        (module_idx == 0 ? device_cluster_prefix_sum[module_idx]
                         : device_cluster_prefix_sum[module_idx] -
                               device_cluster_prefix_sum[module_idx - 1]);

    // Push back the cells to the correct item vector indicated
    // by the cluster prefix sum  -
   
    
    if (cindex < n_clusters)
    {
        //ii = atomicAdd(&cluster_index_atomic[cluster_indice], 1);
      vecmem::device_atomic_ref<unsigned int>(
            cluster_index_atomic[cluster_indice])
            .fetch_add(1);
            
      clusters_device[cluster_indice ][(cluster_index_atomic[cluster_indice])].channel0 = ch0[globalIndex];
      clusters_device[cluster_indice ][(cluster_index_atomic[cluster_indice])].channel1 = ch1[globalIndex];
      clusters_device[cluster_indice ][(cluster_index_atomic[cluster_indice])].activation = activation[globalIndex];
    }
}

}  // namespace traccc::device
