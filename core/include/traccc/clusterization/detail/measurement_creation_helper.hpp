/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"

namespace traccc::detail {

/// Function used for retrieving the cell signal based on the module id
TRACCC_HOST_DEVICE
inline scalar signal_cell_modelling(scalar signal_in,
                                    const cell_module& /*module*/) {
    return signal_in;
}

/// Function for pixel segmentation
TRACCC_DEVICE
inline vector2 position_from_cell(
    const std::size_t channel0,
    const std::size_t channel1,
    const cell_module& module) {
    // Retrieve the specific values based on module idx
    return {module.pixel.min_center_x + channel0 * module.pixel.pitch_x,
            module.pixel.min_center_y + channel1 * module.pixel.pitch_y};
}
/// Function for pixel segmentation
TRACCC_HOST
inline vector2 position_from_cell(const cell& c, const cell_module& module) {
    // Retrieve the specific values based on module idx
    return {module.pixel.min_center_x + c.channel0 * module.pixel.pitch_x,
            module.pixel.min_center_y + c.channel1 * module.pixel.pitch_y};
}

/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[in] cluster The vector of cells describing the identified cluster
/// @param[in] module  The cell module
/// @param[out] mean   The mean position of the cluster/measurement
/// @param[out] var    The variation on the mean position of the
///                    cluster/measurement
/// @param[out] totalWeight The total weight of the cluster/measurement
///


template < typename VV , typename SS >
TRACCC_DEVICE
 inline void calc_cluster_properties(
    VV clusters_device,
    const std::size_t idx_cluster, 
    SS activation,
    VV channel0,
    VV channel1,
    const std::size_t nbr_cell_per_cluster,
    const cell_module& module, point2& mean,
    point2& var, scalar& totalWeight) {

    // Loop over the cells of the cluster.
    for (unsigned int i = 0; i < nbr_cell_per_cluster ; i++ ) {
        // obtenir l'indice global de cell 
        unsigned int cell_index = clusters_device[idx_cluster + i];
        // Translate the cell readout value into a weight.
        const scalar weight = signal_cell_modelling(activation[cell_index], module);

        // Only consider cells over a minimum threshold.
        if (weight > module.threshold) {

            // Update all output properties with this cell.
            totalWeight += activation[cell_index];
            const point2 cell_position = position_from_cell(channel0[cell_index], channel1[cell_index], module);
            const point2 prev = mean;
            const point2 diff = cell_position - prev;

            mean = prev + (weight / totalWeight) * diff;
            for (std::size_t i = 0; i < 2; ++i) {
                var[i] =
                    var[i] + weight * (diff[i]) * (cell_position[i] - mean[i]);
            }
        }
    }
}
template <typename cell_collection_t>
TRACCC_HOST inline void calc_cluster_properties(
    const cell_collection_t& cluster, const cell_module& module, point2& mean,
    point2& var, scalar& totalWeight) {

    // Loop over the cells of the cluster.
    for (const cell& cell : cluster) {

        // Translate the cell readout value into a weight.
        const scalar weight = signal_cell_modelling(cell.activation, module);

        // Only consider cells over a minimum threshold.
        if (weight > module.threshold) {

            // Update all output properties with this cell.
            totalWeight += cell.activation;
            const point2 cell_position = position_from_cell(cell, module);
            const point2 prev = mean;
            const point2 diff = cell_position - prev;

            mean = prev + (weight / totalWeight) * diff;
            for (std::size_t i = 0; i < 2; ++i) {
                var[i] =
                    var[i] + weight * (diff[i]) * (cell_position[i] - mean[i]);
            }
        }
    }
}

/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[out] measurements is the measurement container where the measurement
/// object will be filled
/// @param[in] cluster is the input cell vector
/// @param[in] module is the cell module where the cluster belongs to
/// @param[in] module_link is the module index of the cell container
/// @param[in] cluster_link is the cluster index of the cluster container
///



template <typename measurement_container_t, typename VV , typename SS>
TRACCC_DEVICE inline void fill_measurement(
    measurement_container_t& measurements, 
    VV clusters_device,
     std::size_t idx_cluster,//indice de cluster dans clusters view 
     std::size_t nbr_cell_per_cluster,
    SS activation,
    VV channel0,
    VV channel1,
     cell_module& module, 
      std::size_t module_link,
     std::size_t cl_link /*global index*/) {

    // To calculate the mean and variance with high numerical stability
    // we use a weighted variant of Welford's algorithm. This is a
    // single-pass online algorithm that works well for large numbers
    // of samples, as well as samples with very high values.
    //
    // To learn more about this algorithm please refer to:
    // [1] https://doi.org/10.1080/00401706.1962.10490022
    // [2] The Art of Computer Programming, Donald E. Knuth, second
    //     edition, chapter 4.2.2.

    // Calculate the cluster properties
    scalar totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.};
    detail::calc_cluster_properties(clusters_device, idx_cluster,  nbr_cell_per_cluster,
     activation , channel0, channel1, module, mean, var, totalWeight);

    if (totalWeight > 0.) {
        measurement m;
        // cluster link
        m.cluster_link = cl_link;
        // normalize the cell position
        m.local = mean;
        // normalize the variance
        m.variance[0] = var[0] / totalWeight;
        m.variance[1] = var[1] / totalWeight;
        // plus pitch^2 / 12
        const auto pitch = module.pixel.get_pitch();
        m.variance = m.variance +
                     point2{pitch[0] * pitch[0] / 12, pitch[1] * pitch[1] / 12};
        // @todo add variance estimation

        measurements[module_link].header = module;
        measurements[module_link].items.push_back(std::move(m));
    }
}




/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[out] measurements is the measurement container where the measurement
/// object will be filled
/// @param[in] cluster is the input cell vector
/// @param[in] module is the cell module where the cluster belongs to
/// @param[in] module_link is the module index of the cell container
/// @param[in] cluster_link is the cluster index of the cluster container
///
template <typename measurement_container_t, typename cell_collection_t>
TRACCC_HOST inline void fill_measurement(
    measurement_container_t& measurements, const cell_collection_t& cluster,
    const cell_module& module, const std::size_t module_link,
    const std::size_t cl_link) {

    // To calculate the mean and variance with high numerical stability
    // we use a weighted variant of Welford's algorithm. This is a
    // single-pass online algorithm that works well for large numbers
    // of samples, as well as samples with very high values.
    //
    // To learn more about this algorithm please refer to:
    // [1] https://doi.org/10.1080/00401706.1962.10490022
    // [2] The Art of Computer Programming, Donald E. Knuth, second
    //     edition, chapter 4.2.2.

    // Calculate the cluster properties
    scalar totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.};
    detail::calc_cluster_properties(cluster, module, mean, var, totalWeight);

    if (totalWeight > 0.) {
        measurement m;
        // cluster link
        m.cluster_link = cl_link;
        // normalize the cell position
        m.local = mean;
        // normalize the variance
        m.variance[0] = var[0] / totalWeight;
        m.variance[1] = var[1] / totalWeight;
        // plus pitch^2 / 12
        const auto pitch = module.pixel.get_pitch();
        m.variance = m.variance +
                     point2{pitch[0] * pitch[0] / 12, pitch[1] * pitch[1] / 12};
        // @todo add variance estimation

        measurements[module_link].header = module;
        measurements[module_link].items.push_back(std::move(m));
    }
}


}  // namespace traccc::detail

