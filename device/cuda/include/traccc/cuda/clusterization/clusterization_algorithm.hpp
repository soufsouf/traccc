/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/cuda/utils/stream.hpp"

// Project include(s).
#include "traccc/edm/alt_cell.hpp"
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::cuda {

/// Algorithm performing hit clusterization in a naive way
///
/// This algorithm implements a very trivial parallelization for the hit
/// clusterization. Simply handling every detector module in its own thread.
/// Which is a fairly simple way of translating the single-threaded CPU
/// algorithm, but also a pretty bad algorithm for a GPU.
///
class clusterization_algorithm
    : public algorithm<std::pair<spacepoint_collection_types::buffer,
                                 vecmem::data::vector_buffer<unsigned int>>(
          const alt_cell_collection_types::const_view&,
          const cell_module_collection_types::const_view&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    /// @param target_cells_per_partition the average number of cells in each
    /// partition
    ///
    clusterization_algorithm(const traccc::memory_resource& mr,
                             vecmem::copy& copy, stream& str,
                             const unsigned short target_cells_per_partition);

    /// Callable operator for clusterization algorithm
    ///
    /// @param cells        a collection of cells
    /// @param modules      a collection of modules
    /// @return a spacepoint collection (buffer) and a collection (buffer) of
    /// links from cells to the spacepoints they belong to.
    output_type operator()(
        const alt_cell_collection_types::const_view& cells,
        const cell_module_collection_types::const_view& modules) const override;

    private:
    /// The average number of cells in each partition
    unsigned short m_target_cells_per_partition;
    /// The memory resource(s) to use
    traccc::memory_resource m_mr;
    /// The copy object to use
    vecmem::copy& m_copy;
    /// The CUDA stream to use
    stream& m_stream;
};
class clusterization_algorithm2
    : public algorithm<std::pair<traccc::spacepoint_container,vecmem::data::vector_buffer<unsigned int>>(
          const alt_cell_collection_types::const_view&,
          const cell_module_collection_types::const_view&)>{

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    /// @param target_cells_per_partition the average number of cells in each
    /// partition
    ///
    clusterization_algorithm2(const traccc::memory_resource& mr,
                             vecmem::copy& copy, stream& str,
                             const unsigned short target_cells_per_partition);

    /// Callable operator for clusterization algorithm
    ///
    /// @param cells        a collection of cells
    /// @param modules      a collection of modules
    /// @return a spacepoint collection (buffer) and a collection (buffer) of
    /// links from cells to the spacepoints they belong to.
    output_type operator()(
        const alt_cell_collection_types::const_view& cells,
        const cell_module_collection_types::const_view& modules,
        const traccc::spacepoint_container& spacepoints_container) const override;

    private:
    /// The average number of cells in each partition
    unsigned short m_target_cells_per_partition;
    /// The memory resource(s) to use
    traccc::memory_resource m_mr;
    /// The copy object to use
    vecmem::copy& m_copy;
    /// The CUDA stream to use
    stream& m_stream;
};

}  // namespace traccc::cuda