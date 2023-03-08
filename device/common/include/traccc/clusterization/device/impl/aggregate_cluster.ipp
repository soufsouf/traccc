/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include <vecmem/memory/device_atomic_ref.hpp>
#include "traccc/edm/cell.hpp"
#include "traccc/clusterization/detail/measurement_creation_helper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
namespace traccc::device {

TRACCC_HOST_DEVICE
inline void aggregate_cluster(
    const alt_cell_collection_types::const_device& cells,
    const cell_module_collection_types::const_device& modules,
    const unsigned int start,grp_cluster* cluster_group, const unsigned short cid,
    alt_measurement& out) {

    

    /*
     * Now, we iterate over all other cells to check if they belong
     * to our cluster. Note that we can start at the current index
     * because no cell is ever a child of a cluster owned by a cell
     * with a higher ID.
     */
    float totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.};
     
    const auto module_link = cells[cluster_group[cid].pos].module_link;
    const cell_module this_module = modules.at(module_link);
    index_t id =cluster_group[cid].id_cluster;
    int i = 0 ;
     while(cluster_group[cid + i].id_cluster == id ) 
    {
            const cell this_cell = cells[cluster_group[cid + i].pos].c;
            const float weight = traccc::detail::signal_cell_modelling(
                        this_cell.activation, this_module);
            if (weight > this_module.threshold) {
                        totalWeight += this_cell.activation;
                        const point2 cell_position =
                            traccc::detail::position_from_cell(this_cell, this_module);
                        const point2 prev = mean;
                        const point2 diff = cell_position - prev;

                        mean = prev + (weight / totalWeight) * diff;
                        for (char i = 0; i < 2; ++i) {
                            var[i] = var[i] +
                                    weight * (diff[i]) * (cell_position[i] - mean[i]);
                        }
                    }
        i++;
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
    out.local = mean;
    out.variance = var;
    out.module_link = module_link;
}

}  // namespace traccc::device