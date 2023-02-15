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
    vecmem::data::vector_view<point3 >& global_spacepoint){
//printf("hello world ****** \n");
    vecmem::device_vector<transform3> placement_device(headersView.placement);
    //printf("hello  ****** \n");
    vecmem::device_vector<unsigned int> Cl_module_link(Clusters_module_link);
    vecmem::device_vector<point2> local_measurement(measurement_local);
    vecmem::device_vector<point3> global(global_spacepoint);

    // Ignore if idx is out of range
    if (globalIndex >= Cl_module_link.size())
        return;


/*********************************************************************************/
    // Initialize the rest of the device containers
   
   

   
    /*********************************************************************************/
    
    /*********************************************************************************/
    const auto module_index = Cl_module_link[globalIndex];
    point2 local =  local_measurement[globalIndex];
    //const auto& placement = placement_device[module_index];
    point3 local_3d = {local[0], local[1], 0.};

    if (globalIndex> 1111 && globalIndex < 1200) { printf("local_measurement[globalIndex] %llu\n", local_measurement[globalIndex].at(0));
     }
    //printf("maissa \n");
    global[globalIndex] = placement_device[module_index].point_to_global(local_3d);
    //printf("maissa  2\n");

    
}

}  // namespace traccc::device
