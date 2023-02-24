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
    cluster_container_types::const_view clusters_view,
    const traccc::HeadersView &headersView,
    vecmem::data::vector_view<unsigned int >& Clusters_module_link,
    vecmem::data::vector_view<point2 > &measurement_local,
    vecmem::data::vector_view<point2 >& measurement_variance) {

    // Initialize device vector that gives us the execution range
    
    const cluster_container_types::const_device clusters_device(clusters_view);
    vecmem::device_vector<pixel_data> pixel_device(headersView.pixel);
    vecmem::device_vector<scalar> threshold_device(headersView.threshold);
    vecmem::device_vector<unsigned int> Cl_module_link(Clusters_module_link);
    vecmem::device_vector<point2> local_measurement(measurement_local);
    vecmem::device_vector<point2> variance_measurement(measurement_variance);
    

    // Ignore if idx is out of range
    if (globalIndex >= Cl_module_link.size()) /// faux 
        return;

    // Create other device containers
    

    // items: cluster of cells at current idx
    // header: module idx
    //obtenir les cells de cluster: remplacer par deux vec: 
    //on met dans le premier l'indice de debut des cells d'un cluster dans le vecteur device_clusters 
    //et dans le deuxieme prefix sum on peut obtenir le nombre de cells par cluster 
    
    const auto& cluster = clusters_device[globalIndex].items;
    const auto& module_link = clusters_device[globalIndex].header;
    Cl_module_link[globalIndex] = module_link;
    const scalar threshold = threshold_device[module_link] ;
    const pixel_data pixels = pixel_device[module_link] ;
    
    
    //if (globalIndex > 100 && globalIndex < 105) { printf("cluster.at(1) %llu \n", cluster.at(1).activation); }

    // Should not happen
    //assert(clusters_device.empty() == false);
   
    // Fill measurement from cluster
    if ( globalIndex = 1110  ){
       for (const cell& cell : cluster){
    printf("cell.channel0 %llu", cell.channel0);
  } 
  //} 
   
    detail::fill_measurement(local_measurement,variance_measurement, cluster,  threshold,pixels
          , module_link, globalIndex );

        
    
}           ////// local_measurement and variance_measurement is 0 

}  // namespace traccc::device
