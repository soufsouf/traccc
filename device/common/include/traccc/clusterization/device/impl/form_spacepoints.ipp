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
     vecmem::data::vector_view<point2 > &measurement_local,
      vecmem::data::vector_view<point2 >& measurement_variance,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view,
    spacepoint_container_types::view spacepoints_view) {

    // Initialize device container for for the prefix sum
    vecmem::device_vector<const device::prefix_sum_element_t>
        measurements_prefix_sum(measurements_prefix_sum_view);
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
    measurement_container_types::const_device measurements_device(
        measurements_view);
    spacepoint_container_types::device spacepoints_device(spacepoints_view);

    // Get the indices from the prefix sum vector
    const auto module_idx = measurements_prefix_sum[globalIndex].first;
    const auto measurement_idx = measurements_prefix_sum[globalIndex].second;

    // Get the measurement for this idx
    const auto& m = measurements_device[module_idx].items.at(measurement_idx);

    // Get the current cell module
    const auto& module = measurements_device[module_idx].header;

    // Form a spacepoint based on this measurement
    point3 local_3d = {m.local[0], m.local[1], 0.};
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
    spacepoints_device[module_index].items.push_back(s);
}

}  // namespace traccc::device
