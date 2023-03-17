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
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// VecMem include(s).
#include <vecmem/utils/copy.hpp>
using namespace traccc ;
namespace traccc {
using scalar = TRACCC_CUSTOM_SCALARTYPE;
using uint_collection_types = collection_types<unsigned int>;
using scalar_collection_types = collection_types<scalar>;

struct CellsBuffer {
    uint_collection_types::buffer   channel0;
    uint_collection_types::buffer   channel1;
    scalar_collection_types::buffer activation;
    scalar_collection_types::buffer time;
    uint_collection_types::buffer   module_link;
    std::size_t size;

    CellsBuffer() : size(0) {}

    void SetSize(std::size_t s, vecmem::memory_resource& mr,
                 vecmem::cuda::copy& copy) {
        size = s;
        channel0    = uint_collection_types::buffer(s, mr);
        channel1    = uint_collection_types::buffer(s, mr);
        activation  = scalar_collection_types::buffer(s, mr);
        time        = scalar_collection_types::buffer(s, mr);
        module_link = uint_collection_types::buffer(s, mr);
        copy.setup(channel0);
        copy.setup(channel1);
        copy.setup(activation);
        copy.setup(time);
        copy.setup(module_link);
    }

    void CopyToDevice(const CellsHost &c,
                      vecmem::cuda::copy& copy) {
        copy(vecmem::get_data(c.channel0), channel0,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(c.channel1), channel1,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(c.activation), activation,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(c.time), time,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(c.module_link), module_link,
             vecmem::copy::type::copy_type::host_to_device);
    }
};

struct CellsView {
    uint_collection_types::view   channel0;
    uint_collection_types::view   channel1;
    scalar_collection_types::view activation;
    scalar_collection_types::view time;
    uint_collection_types::view   module_link;
    std::size_t size;

    CellsView() = delete;
    CellsView(const traccc::CellsBuffer &c) {
        channel0    = c.channel0;
        channel1    = c.channel1;
        activation  = c.activation;
        time        = c.time;
        module_link = c.module_link;
        size = c.size;
    }

    CellsView(const CellsView &c) {
        channel0    = c.channel0;
        channel1    = c.channel1;
        activation  = c.activation;
        time        = c.time;
        module_link = c.module_link;
        size = c.size;
    }
};

struct CellsRefDevice {
    uint_collection_types::device   channel0;
    uint_collection_types::device   channel1;
    scalar_collection_types::device activation;
    scalar_collection_types::device time;
    uint_collection_types::device   module_link;
    CellsRefDevice() = delete;
TRACCC_DEVICE
    CellsRefDevice(const traccc::CellsView &c)
    : channel0{c.channel0},
      channel1(c.channel1),
      activation(c.activation),
      time(c.time),
      module_link(c.module_link) {}
};
}

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
class clusterization_algorithm3
    : public algorithm<std::pair<spacepoint_collection_types::buffer,
                                 vecmem::data::vector_buffer<unsigned int>>(
          const alt_cell_collection_types::const_view&,
          const cell_module_collection_types::const_view&,
          const traccc::CellsView&)> {

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
    clusterization_algorithm3(const traccc::memory_resource& mr,
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
        const traccc::CellsView& cellsSoA) const override;

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
    : public algorithm<std::pair<spacepoint_collection_types::buffer,
                                 vecmem::data::vector_buffer<unsigned int>>(
          const alt_cell_collection_types::const_view&,
          const cell_module_collection_types::const_view&,
          const traccc::CellsView&)> {

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
        const traccc::CellsView& cellsSoA) const override;

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