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
<<<<<<< HEAD
     vecmem::data::vector_view<point2 >& measurement_local,
      vecmem::data::vector_view<point2 >& measurement_variance,
    spacepoint_container_types::view spacepoints_view) {

    // Initialize device container for for the prefix sum
   
=======
     vecmem::data::vector_view<point2 > &measurement_local,
      vecmem::data::vector_view<point2 >& measurement_variance,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view,
    spacepoint_container_types::view spacepoints_view) {

    // Initialize device container for for the prefix sum
    vecmem::device_vector<const device::prefix_sum_element_t>
        measurements_prefix_sum(measurements_prefix_sum_view);
>>>>>>> 75b755426e95a3e36f94b52f6f64689c492efbce
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
<<<<<<< HEAD
    point3 global = module_.placement.point_to_global(local_3d);

    spacepoint s({global, m});

    // Push the speacpoint into the container at the appropriate
    // module idx
    spacepoints_device[module_index].header = module.module;
=======
    point3 global = module.placement.point_to_global(local_3d);
    /*********************************************************************************/
    const auto module_index = Cl_module_link[globalIndex];
    measurement mm;
    mm.local =  local_measurement[globalIndex];
    mm.cluster_link = globalIndex;   
    mm.variance = variance_measurement[globalIndex];
    const auto& module_ = cells_device.at(module_index).header;
    point3 local__3d = {mm.local[0], mm.local[1], 0.};
    point3 global_ = module_.placement.point_to_global(local__3d);

    spacepoint s({global_, mm});

    // Push the speacpoint into the container at the appropriate
    // module idx
    spacepoints_device[module_index].header = module_.module;
>>>>>>> 75b755426e95a3e36f94b52f6f64689c492efbce
    spacepoints_device[module_index].items.push_back(s);
}

}  // namespace traccc::device
