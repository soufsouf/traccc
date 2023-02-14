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
    vecmem::data::vector_view<unsigned int > celllabel,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,   /// not used 
    const CellView& cellView,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view
    )   {

    // Get the device vector of the cell prefix sum
   // printf("hello yes \n");
    vecmem::device_vector<unsigned int> midx(cellView.module_id);
       //if(globalIndex < 10) printf("hello no \n");
     //printf(" midx count cluster est %u \n", midx[globalIndex]);
    vecmem::device_vector<unsigned int> labels(celllabel);
    
    //printf(" label count cluster est %u \n", labels[globalIndex]);
   

    // Ignore if idx is out of range
    if (globalIndex >= labels.size())
        return;

    // Get the indices for the module and the cell in this
    // module, from the prefix sum
   auto module_idx = midx[globalIndex];
    //printf(" hello maissa 1\n");
   unsigned int cindex = labels[globalIndex] - 1;
    // Vectors used for cluster indices found by sparse CCL
   //printf(" label count cluster est %u \n", cindex);
    
    // Get the cluster prefix sum at this module_idx to know
    // where to write current clusters in the
    // cluster container
    vecmem::device_vector<std::size_t> device_cluster_prefix_sum(
        cluster_prefix_sum_view);
     
    std::size_t prefix_sum =
        (module_idx == 0 ? 0 : device_cluster_prefix_sum[module_idx - 1]);
    std::size_t cluster_indice = prefix_sum + cindex;  
    // Calculate the number of clusters found for this module from the prefix
    // sums
    std::size_t n_clusters =
        (module_idx == 0 ? device_cluster_prefix_sum[0]
                         : device_cluster_prefix_sum[module_idx] - prefix_sum);
     printf(" hello 1 %llu , hello 2 %u , prefix sum %llu   device_cluster_prefix_sum[]: %llu  \n",n_clusters,cindex,prefix_sum, device_cluster_prefix_sum[module_idx] );
    // Vector to fill in with the sizes of each cluster
    vecmem::device_vector<unsigned int> device_cluster_sizes(
        cluster_sizes_view);

    // Count the cluster sizes for each position
   
    if (cindex < n_clusters) {
        atomicAdd(&device_cluster_sizes[cluster_indice], 1);
        
        /*vecmem::device_atomic_ref<unsigned int>(
            device_cluster_sizes[cluster_indice])
            .fetch_add(1);*/
    }
printf("fin count cluster");
    }  
    
   // __syncthreads();
    // brust prefix sum (scan operation)
    
   // thrust::inclusive_scan(thrust::device , device_cluster_sizes.begin(), device_cluster_sizes.end() , cells_cluster_prefix_sum.begin()); // in-place scan
   

    /*if(globalIndex == 0)
    {
        cells_cluster_prefix_sum[0] = device_cluster_sizes[0];
        for(unsigned int i = 1; i < device_cluster_sizes.size()   ; i++)
        {
            cells_cluster_prefix_sum[i] = device_cluster_sizes[i ] + cells_cluster_prefix_sum[i - 1];
        }
    }*/
    

 //__syncthreads();

   /* if (globalIndex < 64) {
        printf("cl_size %u cluster_prefix_sum %u \n", 
                device_cluster_sizes[cluster_indice], cells_cluster_prefix_sum[globalIndex]); } */
    

///// connect components : 

/*unsigned int idx = 
        (cluster_indice == 0 ? 0 : cluster_indice - 1);
    unsigned int lb = cells_cluster_prefix_sum[idx];
    
    unsigned int ii = 0;*/
    //if (cindex < n_clusters)

       // ii = atomicAdd(&cluster_index_atomic[cluster_indice], 1);
        /*vecmem::device_atomic_ref<unsigned int>(
            cluster_index_atomic[cluster_indice])
            .fetch_add(1);*/
       // clusters_device[ii +lb] = globalIndex;
       

}  // namespace traccc::device
