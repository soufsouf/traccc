/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include <vecmem/memory/device_atomic_ref.hpp>

#include "traccc/clusterization/detail/measurement_creation_helper.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void aggregate_cluster(
    const cell_module_collection_types::const_device& modules,
    cluster* id_fathers,
    const unsigned int start, const unsigned int end, const unsigned short cid,
    spacepoint_collection_types::device spacepoints_device,
     vecmem::data::vector_view<unsigned int> cell_links,
    const unsigned int link) {

    //const vecmem::device_vector<unsigned short> f(f_view);
    vecmem::device_vector<unsigned int> cell_links_device(cell_links);
 //spacepoint_collection_types::device spacepoints_device(spacepoints_view);
    /*
     * Now, we iterate over all other cells to check if they belong
     * to our cluster. Note that we can start at the current index
     * because no cell is ever a child of a cluster owned by a cell
     * with a higher ID.
     */
    float totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.};
    const auto module_link = id_fathers[cid].module_link;
    const cell_module this_module = modules.at(module_link);
    const unsigned short partition_size = end - start;

    channel_id maxChannel1 = std::numeric_limits<channel_id>::min();

    for (unsigned short j = cid; j < partition_size; j++) {

        assert(j < id_fathers.size());

        const unsigned int pos = j + start;
        /*
         * Terminate the process earlier if we have reached a cell sufficiently
         * in a different module.
         */
        if (id_fathers[cid].module_link != module_link) {
            break;
        }

        const channel_id c0 = id_fathers[j].channel0;
        const channel_id c1 = id_fathers[j].channel1;
        const scalar activation = id_fathers[j].activation;

        if (id_fathers[j].id_cluster == cid) {

            if (c1 > maxChannel1) {
                maxChannel1 = c1;
            }

            const float weight = traccc::detail::signal_cell_modelling(
                activation, this_module);

            if (weight > this_module.threshold) {
                totalWeight += activation;
                const point2 cell_position =
                    traccc::detail::position_from_cell(c0,c1, this_module);
                const point2 prev = mean;
                const point2 diff = cell_position - prev;

                mean = prev + (weight / totalWeight) * diff;
                for (char i = 0; i < 2; ++i) {
                    var[i] = var[i] +
                             weight * (diff[i]) * (cell_position[i] - mean[i]);
                }
            }

            cell_links_device.at(pos) = link;
        }

        /*
         * Terminate the process earlier if we have reached a cell sufficiently
         * far away from the cluster in the dominant axis.
         */
        if (c1 > maxChannel1 + 1) {
            break;
        }
    }
    if (totalWeight > 0.) {
        for (char i = 0; i < 2; ++i) {
            var[i] /= totalWeight;
        }
        const auto pitch = this_module.pixel.get_pitch();
        var = var + point2{pitch[0] * pitch[0] / 12, pitch[1] * pitch[1] / 12};
    }

    /*
     * Fill output vector with calculated cluster properties
     */
   

    point3 local_3d = {mean[0], mean[1], 0.};
    point3 global = this_module.placement.point_to_global(local_3d);

    // Fill the result object with this spacepoint
    spacepoints_device[link] = {global, {mean, var, 0}};
}

}  // namespace traccc::device