// namespace traccc
/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"
// VecMem include(s).
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/cuda/copy.hpp"

namespace traccc {

/// Alternative cell structure which contains both the cell and a link to its
/// module (held in a separate collection).
///
/// This can be used for storing all information in a single collection, whose
/// objects need to have both the header and item information from the cell
/// container types
struct alt_cell {
    cell c;
    using link_type = cell_module_collection_types::view::size_type;
    link_type module_link;
};

using scalar = TRACCC_CUSTOM_SCALARTYPE;
using uint_collection_types = collection_types<unsigned int>;
using scalar_collection_types = collection_types<scalar>;

struct CellsHost {
    uint_collection_types::host   channel0;
    uint_collection_types::host   channel1;
    scalar_collection_types::host activation;
    scalar_collection_types::host time;
    uint_collection_types::host   module_link;
    std::size_t size;

    CellsHost() : size(0) {}

    void SetSize(std::size_t s, vecmem::memory_resource *mr) {
        size = s;
        channel0    = uint_collection_types::host(s, mr);
        channel1    = uint_collection_types::host(s, mr);
        activation  = scalar_collection_types::host(s, mr);
        time        = scalar_collection_types::host(s, mr);
        module_link = uint_collection_types::host(s, mr);
    }
};

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
struct cluster{
    unsigned short id_cluster ;
    channel_id channel0 = 0;
    channel_id channel1 = 0;
    scalar activation = 0.;
    using link_type = cell_module_collection_types::view::size_type;
    link_type module_link;
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

/// Declare all cell collection types
using alt_cell_collection_types = collection_types<alt_cell>;

/// Type definition for the reading of cells into a vector of alt_cells and a
/// vector of modules. The alt_cells hold a link to a position in the modules'
/// vector.
struct alt_cell_reader_output_t {
    alt_cell_collection_types::host cells;
    cell_module_collection_types::host modules;
};

}  // namespace traccc