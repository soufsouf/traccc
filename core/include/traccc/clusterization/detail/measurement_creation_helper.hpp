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
TRACCC_HOST
inline scalar signal_cell_modelling(scalar signal_in,
                                    const cell_module& module) {
    return signal_in;
}
TRACCC_DEVICE
inline scalar signal_cell_modelling(scalar signal_in) {
    return signal_in;
}

/// Function for pixel segmentation
TRACCC_HOST
inline vector2 position_from_cell(const cell& c, const cell_module& module) {
    // Retrieve the specific values based on module idx
    return {module.pixel.min_center_x + c.channel0 * module.pixel.pitch_x,
            module.pixel.min_center_y + c.channel1 * module.pixel.pitch_y};
}
TRACCC_DEVICE
inline vector2 position_from_cell(const cell& c,const pixel_data pixels ) {
    // Retrieve the specific values based on module idx
    return {pixels.min_center_x + c.channel0 * pixels.pitch_x,
            pixels.min_center_y + c.channel1 * pixels.pitch_y};
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


template <typename cell_collection_t>
TRACCC_HOST
void calc_cluster_properties(
    const cell_collection_t& cluster,
    const cell_module& module, point2& mean,
    point2& var, scalar& totalWeight , const std::size_t cl_link) {

    // Loop over the cells of the cluster.
    
     for (const cell& cell : cluster) {

        // Translate the cell readout value into a weight.
        const scalar weight = signal_cell_modelling(cell.activation, module);
        
      /// print 
    
       // printf("weight   %llu module.threshold   %llu\n", totalWeight , module.threshold );
                 

        // Only consider cells over a minimum threshold.
        if (weight > module.threshold) {

            // Update all output properties with this cell.
            totalWeight += cell.activation;
            const point2 cell_position = position_from_cell(cell, module);
            const point2 prev = mean;
            const point2 diff = cell_position - prev;

            mean = prev + (weight / totalWeight) * diff;
            for (unsigned int j = 0; j < 2; ++j) {
                var[j] =
                    var[j] + weight * (diff[j]) * (cell_position[j] - mean[j]);
            }
        }
    }
   
}
template <typename cell_collection_t>
TRACCC_DEVICE
void calc_cluster_properties(
    const cell_collection_t& cluster,
    const scalar threshold,
    const pixel_data pixels,
     point2& mean,
    point2& var, scalar& totalWeight ,
    const std::size_t cl_link) {

    // Loop over the cells of the cluster.

    for (const cell& cell : cluster) {
    
        // Translate the cell readout value into a weight.
        const scalar weight = cell.activation ; 

         

        // Only consider cells over a minimum threshold.
        if (weight > threshold) {

            // Update all output properties with this cell.
            totalWeight += cell.activation;
            const point2 cell_position = position_from_cell(cell, pixels);
            const point2 prev = mean;
            const point2 diff = cell_position - prev;

            mean = prev + (weight / totalWeight) * diff;
            for (unsigned int j = 0; j < 2; ++j) {
                var[j] =
                    var[j] + weight * (diff[j]) * (cell_position[j] - mean[j]);
            }
        }
    }
   
    if (cl_link > 1111 ) {
    printf("var[0] %llu var[1] %llu mean[0] %llu mean[1] %llu \n",
            var[0] , var[1] , mean[0] , mean[1]); } 
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



template <typename PP , typename cell_collection_t>
TRACCC_DEVICE inline void fill_measurement( 
    PP& local_measurement, 
    PP& variance_measurement,
    const cell_collection_t& cluster,
    const scalar threshold,
    const pixel_data pixels, 
    const std::size_t module_link,
    const std::size_t cl_link) {

    // Calculate the cluster properties
    scalar totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.}, variance{0., 0.};
    detail::calc_cluster_properties(cluster, threshold,pixels, mean, var, totalWeight ,cl_link );

     //printf("threshold %lf \n",threshold);
    if (totalWeight > threshold)
    {
        // cluster link
        // normalize the cell position   
        local_measurement[cl_link]= mean;
        // normalize the variance
        variance[0]=var[0] / totalWeight;
        variance[1] = var[1] / totalWeight;
        // plus pitch^2 / 12
        const auto pitch = pixels.get_pitch();
        // @todo add variance estimation
        variance_measurement[cl_link]= variance + point2{pitch[0] * pitch[0] / 12, pitch[1] * pitch[1] / 12};
        /*printf("th %llu totweight %lf module %llu[%lf] pitch[%lf, %lf] \n", cl_link, totalWeight,
            module_link, module.threshold, pitch[0], pitch[1]);*/

           /* if (cl_link > 9000 &&  cl_link < 9004) 
            printf("var[0] %llu var[1] %llu\n",
             var[0] , var[1]);   */
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
    detail::calc_cluster_properties(cluster, module, mean, var, totalWeight , cl_link);

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
