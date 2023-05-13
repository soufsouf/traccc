/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/geometry/pixel_data.hpp"
// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <cmath>
#include <limits>

namespace traccc {

/// Definition for one detector cell
///
/// It comes with two integer channel identifiers, an "activation value"
/// and a time stamp.
///
struct cell {
    channel_id channel0 = 0;
    channel_id channel1 = 0;
    scalar activation = 0.;
    scalar time = 0.;
};

/// Comparison / ordering operator for cells
TRACCC_HOST_DEVICE
inline bool operator<(const cell& lhs, const cell& rhs) {

    if (lhs.channel0 != rhs.channel0) {
        return (lhs.channel0 < rhs.channel0);
    } else if (lhs.channel1 != rhs.channel1) {
        return (lhs.channel1 < rhs.channel1);
    } else {
        return lhs.activation < rhs.activation;
    }
}

/// Equality operator for cells
TRACCC_HOST_DEVICE
inline bool operator==(const cell& lhs, const cell& rhs) {

    return (
        (lhs.channel0 == rhs.channel0) && (lhs.channel1 == rhs.channel1) &&
        (std::abs(lhs.activation - rhs.activation) < traccc::float_epsilon) &&
        (std::abs(lhs.time - rhs.time) < traccc::float_epsilon));
}

/// Header information for all of the cells in a specific detector module
///
/// It is handled separately from the list of all of the cells belonging to
/// the detector module, to be able to lay out the data in memory in a way
/// that is more friendly towards accelerators.
///
struct cell_module {

    geometry_id module = 0;
    transform3 placement = transform3{};
    scalar threshold = 0;

    pixel_data pixel;

};  // struct cell_module

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

struct CellsDevice {
    uint_collection_types::device   channel0;
    uint_collection_types::device   channel1;
    scalar_collection_types::device activation;
    scalar_collection_types::device time;
    uint_collection_types::device   module_link;
    
    CellsDevice() = delete;
    TRACCC_HOST_DEVICE
    CellsDevice(const traccc::CellsView &c)
    : channel0(c.channel0),
      channel1(c.channel1),
      activation(c.activation),
      time(c.time),
      module_link(c.module_link) {}
};

/*******************************************************************************/
/*struct Cluster {
    int_view cells_prefix_sum;
    int_view module_id;
};*/

/*******************************************************************************/
struct ModulesHost {
    uint_collection_types::host cells_prefix_sum;
    uint_collection_types::host clusters_prefix_sum;
    std::size_t size;

    ModulesHost() : size(0) {}

    void SetSize(std::size_t s, vecmem::memory_resource *mr) {
        size = s;
        cells_prefix_sum    = uint_collection_types::host(s, mr);
        clusters_prefix_sum = uint_collection_types::host(s, mr);
    }
};

struct ModulesBuffer {
    uint_collection_types::buffer cells_prefix_sum;
    uint_collection_types::buffer clusters_prefix_sum;
    std::size_t size;

    ModulesBuffer() : size(0) {}

    void SetSize(std::size_t s, vecmem::memory_resource& mr,
                 vecmem::cuda::copy& copy) {
        size = s;
        cells_prefix_sum    = uint_collection_types::buffer(s, mr);
        clusters_prefix_sum = uint_collection_types::buffer(s, mr);
        copy.setup(cells_prefix_sum);
        copy.setup(clusters_prefix_sum);
    }

    void CopyToDevice(const ModulesHost &m,
                      vecmem::cuda::copy& copy) {
        copy(vecmem::get_data(m.cells_prefix_sum), cells_prefix_sum,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(m.clusters_prefix_sum), clusters_prefix_sum,
             vecmem::copy::type::copy_type::host_to_device);
    }
};

struct ModulesView {
    uint_collection_types::view cells_prefix_sum;
    uint_collection_types::view clusters_prefix_sum;
    std::size_t size;

    ModulesView() = delete;
    ModulesView(const traccc::ModulesBuffer &m) {
        cells_prefix_sum    = m.cells_prefix_sum;
        clusters_prefix_sum    = m.clusters_prefix_sum;
        size = m.size;
    }

    ModulesView(const ModulesView &m) {
        cells_prefix_sum    = m.cells_prefix_sum;
        clusters_prefix_sum = m.clusters_prefix_sum;
        size = m.size;
    }
};

struct ModulesDevice {
    uint_collection_types::device cells_prefix_sum;
    uint_collection_types::device clusters_prefix_sum;

    ModulesDevice() = delete;
    TRACCC_HOST_DEVICE
    ModulesDevice(const traccc::ModulesView &m)
    : cells_prefix_sum(m.cells_prefix_sum),
      clusters_prefix_sum(m.clusters_prefix_sum) {}
};

/*******************************************************************************/
/// Equality operator for cell module
TRACCC_HOST_DEVICE
inline bool operator==(const cell_module& lhs, const cell_module& rhs) {
    return lhs.module == rhs.module;
}

/// Declare all cell collection types
using cell_collection_types = collection_types<cell>;
/// Declare all cell container types
using cell_container_types = container_types<cell_module, cell>;

}  // namespace traccc
