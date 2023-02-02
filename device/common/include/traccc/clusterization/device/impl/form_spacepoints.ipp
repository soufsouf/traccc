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
      vecmem::data::vector_view<point2 >& measurement_variance,
    spacepoint_container_types::view spacepoints_view) {

    // Initialize device container for for the prefix sum
   
   //////////
    cell_container_types::const_device cells_device(cells_view);
    vecmem::device_vector<unsigned int> Cl_module_link(Clusters_module_link);
    vecmem::device_vector<point2> local_measurement(measurement_local);
    vecmem::device_vector<point2> variance_measurement(measurement_variance);

    // Ignore if idx is out of range
    if (globalIndex >= Cl_module_link.size())
        return;


/*********************************************************************************/
    // Initialize the rest of the device containers
   
    spacepoint_container_types::device spacepoints_device(spacepoints_view);

   
    /*********************************************************************************/
    const auto module_index = Cl_module_link[globalIndex];
    measurement m;
    m.local =  local_measurement[globalIndex];
    m.cluster_link = globalIndex;   
    m.variance = variance_measurement[globalIndex];
    const auto& module = cells_device.at(module_index).header;
    point3 local_3d = {m.local[0], m.local[1], 0.};
    point3 global = module_.placement.point_to_global(local_3d);

    spacepoint s({global, m});

    // Push the speacpoint into the container at the appropriate
    // module idx
    spacepoints_device[module_index].header = module.module;
    spacepoints_device[module_index].items.push_back(s);
}

}  // namespace traccc::device