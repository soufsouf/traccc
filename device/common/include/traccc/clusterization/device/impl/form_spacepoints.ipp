/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void form_spacepoints(
    std::size_t globalIndex,
    const traccc::headerView headersView,
    vecmem::data::vector_view<unsigned int > Clusters_module_link,
    vecmem::data::vector_view<point2 > measurement_local,
    vecmem::data::vector_view<point2 > measurement_variance,
    spacepoint_container_types::view spacepoints_view){

    spacepoint_container_types::device spacepoints_device(spacepoints_view);
    vecmem::device_vector<geometry_id> module_device(headersView.module);
    vecmem::device_vector<transform3> placement_device(headersView.placement);
    vecmem::device_vector<point2> variance_measurement(measurement_variance);
    vecmem::device_vector<unsigned int> Cl_module_link(Clusters_module_link);
    vecmem::device_vector<point2> local_measurement(measurement_local);
    

    // Ignore if idx is out of range
    if (globalIndex >= Cl_module_link.size())
        return;


/*********************************************************************************/
    // Initialize the rest of the device containers
   
   

   
    /*********************************************************************************/
    
    /*********************************************************************************/


    const auto module_link = Cl_module_link[globalIndex];
    const auto local_ = local_measurement[globalIndex];
    const auto variance_ = variance_measurement[globalIndex];
    const point3 local_3d = {local_[0], local_[1], 0.};
    const auto global = placement_device[module_link].point_to_global(local_3d);
    
    measurement m;
    m.cluster_link = module_link;
    m.local = local_;
    m.variance = variance_;

  if ( globalIndex > 9110 && globalIndex < 9120 ){
    printf("\n global[globalIndex] : %llu  , local[0] : %llu , local[1] : %llu \n", global , m.local[0] , m.local[1]);
  } 

  spacepoint s({global, m});
  spacepoints_device[module_link].header = module_device[module_link];
  spacepoints_device[module_link].items.push_back(s);  
}

}  // namespace traccc::device
