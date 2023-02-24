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
    std::size_t globalIndex, 
    vecmem::data::vector_view<unsigned int > moduleidx,
    vecmem::data::vector_view<unsigned int > celllabel,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,//cluster per module
    vecmem::data::vector_view<unsigned int > cluster_atomic,
    vecmem::data::vector_view<unsigned int > cel_cl_ps,
    vecmem::data::vector_view<unsigned int > clusters_view) {

    // Get device vector of the cells prefix sum
    vecmem::device_vector<unsigned int> midx(moduleidx);
    vecmem::device_vector<unsigned int> clusters_device(clusters_view);
    vecmem::device_vector<unsigned int> labels(celllabel);
    vecmem::device_vector<unsigned int> cluster_index_atomic(cluster_atomic);
    vecmem::device_vector<unsigned int> cells_per_cluster_prefix_sum(cel_cl_ps);
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
   
    unsigned int idx = 
        (cluster_indice == 0 ? 0 : cluster_indice - 1);
    unsigned int lb = cells_per_cluster_prefix_sum[idx] ;
    
    if (cindex < n_clusters)
    {
        //ii = atomicAdd(&cluster_index_atomic[cluster_indice], 1);
      vecmem::device_atomic_ref<unsigned int>(
            cluster_index_atomic[cluster_indice])
            .fetch_add(1);
      clusters_device[cluster_index_atomic[cluster_indice] +lb ] = globalIndex;
    }
}

TRACCC_DEVICE
inline void connect_components(
    std::size_t globalIndex,
    const CellsView& cellsView,
    vecmem::data::vector_view<unsigned int> cluster_prefix_sum_view,//cluster per module
    cluster_container_types::view clusters_view) {

    // Get device vector of the cells prefix sum
    vecmem::device_vector<scalar> activation(cellsView.activation);
    vecmem::device_vector<unsigned int> ch0(cellsView.channel0);
    vecmem::device_vector<unsigned int> ch1(cellsView.channel1);
    vecmem::device_vector<unsigned int> midx(cellsView.module_id);
    cluster_container_types::device clusters_device(clusters_view);
    vecmem::device_vector<unsigned int> labels(cellsView.label);
    vecmem::device_vector<unsigned int> device_cluster_prefix_sum(cluster_prefix_sum_view);

    if (globalIndex >= labels.size())
        return;

    // Get the indices for the module idx and the cell idx
    auto module_idx = midx[globalIndex];

    // Get the cells for the current module idx
    

    // Vectors used for cluster indices found by sparse CCL
    unsigned int cindex = labels[globalIndex] - 1 ;
    unsigned int prefix_sum;
    unsigned int cluster_indice;
    // Get the cluster prefix sum for this module idx
    if (module_idx == 0)   cluster_indice = cindex ; 
    if (module_idx != 0) {
        prefix_sum = module_idx - 1;
        cluster_indice = device_cluster_prefix_sum[prefix_sum]+ cindex; 
    }
    // Calculate the number of clusters found for this module from the prefix
    // sums
    const unsigned int n_clusters =
        (module_idx == 0 ? device_cluster_prefix_sum[0]
                         : device_cluster_prefix_sum[module_idx] -
                               device_cluster_prefix_sum[module_idx - 1]);

    if (cindex < n_clusters)
    {

       clusters_device[cluster_indice].header = module_idx; 

             // if (globalIndex > 1011 && globalIndex  < 1020) { printf(" cluster_indice %llu \n", cluster_indice ); }

       clusters_device[cluster_indice].items.push_back({ch0[globalIndex] ,
                ch1[globalIndex] , activation[globalIndex] , 0.  });

                if ( globalIndex > 1111  && globalIndex < 1120 ){
       
    printf("cluster_indice %llu \n  ", cluster_indice  );
   
  }
    }

      // if (globalIndex > 100) { printf(" module_idx  %llu header %llu \n", module_idx,clusters_device[globalIndex].header ); }
 
}

}  // namespace traccc::device
