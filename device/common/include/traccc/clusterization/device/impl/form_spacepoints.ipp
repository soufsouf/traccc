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
    const cell_container_types::const_view& cells_view,
     vecmem::data::vector_view<unsigned int >& Clusters_module_link,
     vecmem::data::vector_view<point2 >& measurement_local,
    vecmem::data::vector_view<point3 >& global_spacepoint){

    cell_container_types::const_device cells_device(cells_view);
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
    const auto& module = cells_device.at(module_index).header;
    point3 local_3d = {local[0], local[1], 0.};
    global[globalIndex] = module.placement.point_to_global(local_3d);

    
}

}  // namespace traccc::device
