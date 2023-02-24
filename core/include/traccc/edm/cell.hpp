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
#include <vecmem/utils/cuda/async_copy.hpp>
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

using uint_vec = vecmem::vector<unsigned int>;
using scalar_vec = vecmem::vector<scalar>;
using geometry_id_vec = vecmem::vector<geometry_id>;
using transform3_vec = vecmem::vector<transform3>;
using pixel_data_vec = vecmem::vector<pixel_data>;

using uint_buf = vecmem::data::vector_buffer<unsigned int>;
using scalar_buf = vecmem::data::vector_buffer<scalar>;
using geometry_id_buf = vecmem::data::vector_buffer<geometry_id>;
using transform3_buf = vecmem::data::vector_buffer<transform3>;
using pixel_data_buf = vecmem::data::vector_buffer<pixel_data>;

using uint_view = vecmem::data::vector_view<unsigned int>;
using scalar_view = vecmem::data::vector_view<scalar>;
using geometry_id_view = vecmem::data::vector_view<geometry_id>;
using transform3_view = vecmem::data::vector_view<transform3>;
using pixel_data_view = vecmem::data::vector_view<pixel_data>;

using uint_device = vecmem::device_vector<unsigned int>;
using scalar_device = vecmem::device_vector<scalar>;

struct HeadersHost{
    geometry_id_vec module;
    transform3_vec  placement;
    scalar_vec      threshold;
    pixel_data_vec  pixel;
    std::size_t size;

    void Resize(std::size_t s) {
        module.resize(s);
        placement.resize(s);
        threshold.resize(s);
        pixel.resize(s);
        size = s;
    }
};

struct HeadersBuff{
    geometry_id_buf module;
    transform3_buf  placement;
    scalar_buf      threshold;
    pixel_data_buf  pixel;
    std::size_t size;

    HeadersBuff()
        : size(0) {
        module    = geometry_id_buf();
        placement = transform3_buf();
        threshold = scalar_buf();
        pixel     = pixel_data_buf();
    }

    void Resize(std::size_t s, vecmem::memory_resource& mr,
                 vecmem::cuda::async_copy& copy) {
        size = s;
        module    = geometry_id_buf(s, mr);
        placement = transform3_buf(s, mr);
        threshold = scalar_buf(s, mr);
        pixel     = pixel_data_buf(s, mr);

        copy.setup(module);
        copy.setup(placement);
        copy.setup(threshold);
        copy.setup(pixel);
    }

    void CopyToDevice(const HeadersHost &headersHost,
                      vecmem::cuda::async_copy& copy) {
        copy(vecmem::get_data(headersHost.module), module,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(headersHost.placement), placement,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(headersHost.threshold), threshold,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(headersHost.pixel), pixel,
             vecmem::copy::type::copy_type::host_to_device);
    }
};

struct HeadersView{
    geometry_id_view module;
    transform3_view  placement;
    scalar_view      threshold;
    pixel_data_view  pixel;
    std::size_t size;

    HeadersView() = delete;

    HeadersView(HeadersBuff &b) {
        module    = b.module;
        placement = b.placement;
        threshold = b.threshold;
        pixel     = b.pixel;
        size = b.size;
    }
};

struct CellsHost {
    uint_vec    channel0;
    uint_vec    channel1;
    scalar_vec  activation;
    scalar_vec  time;
    uint_vec    module_id;
    uint_vec    cluster_id;
    std::size_t size;

    void Resize(std::size_t s) {
        channel0.resize(s);
        channel1.resize(s);
        activation.resize(s);
        time.resize(s);
        module_id.resize(s);
        cluster_id.resize(s);
        size = s;
    }
};

struct CellsBuffer {
    uint_buf   channel0;
    uint_buf   channel1;
    scalar_buf activation;
    scalar_buf time;
    uint_buf   module_id;
    uint_buf   cluster_id;
    uint_buf   label; // needed to find clusters
    std::size_t size;

    CellsBuffer()
        : size(0) {
        channel0   = uint_buf();
        channel1   = uint_buf();
        activation = scalar_buf();
        time       = scalar_buf();
        module_id  = uint_buf();
        cluster_id = uint_buf();
        label = uint_buf();
    }

    void Resize(std::size_t s, vecmem::memory_resource& mr,
                 vecmem::cuda::async_copy& copy) {
        size = s;
        channel0   = uint_buf(s, mr);
        channel1   = uint_buf(s, mr);
        activation = scalar_buf(s, mr);
        time       = scalar_buf(s, mr);
        module_id  = uint_buf(s, mr);
        cluster_id = uint_buf(s, mr);
        label = uint_buf(s, mr);
        copy.setup(channel0);
        copy.setup(channel1);
        copy.setup(activation);
        copy.setup(time);
        copy.setup(module_id);
        copy.setup(cluster_id);
        copy.setup(label);
    }

    void CopyToDevice(const CellsHost &c,
                      vecmem::cuda::async_copy& copy) {
        copy(vecmem::get_data(c.channel0), channel0,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(c.channel1), channel1,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(c.activation), activation,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(c.time), time,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(c.module_id), module_id,
             vecmem::copy::type::copy_type::host_to_device);
        copy(vecmem::get_data(c.cluster_id), cluster_id,
             vecmem::copy::type::copy_type::host_to_device);
        // label is written in device
    }
};

struct CellsView {
    uint_view   channel0;
    uint_view   channel1;
    scalar_view activation;
    scalar_view time;
    uint_view   module_id;
    uint_view   cluster_id;
    uint_view   label;
    std::size_t size;

    CellsView() = delete;
    CellsView(const traccc::CellsBuffer &c) {
        channel0   = c.channel0;
        channel1   = c.channel1;
        activation = c.activation;
        time       = c.time;
        module_id  = c.module_id;
        cluster_id = c.cluster_id;
        label = c. label;
        size = c.size;
    }
};

struct CellsRefDevice {
    uint_device   channel0;
    uint_device   channel1;
    scalar_device activation;
    scalar_device time;
    uint_device   module_id;
    uint_device   cluster_id;
    uint_device   label;
/*
TRACCC_HOST_DEVICE
    CellsRefDevice(const traccc::CellsDevice &c)
        : channel0(int_device(c.channel0)) {
        channel1   = int_device(c.channel1);
        activation = scalar_device(c.activation);
        time       = scalar_device(c.time);
        module_id  = int_device(c.module_id);
        cluster_id = int_device(c.cluster_id);
    }*/ 
};

struct ModulesHost {
    uint_vec cells_prefix_sum;
    uint_vec clusters_prefix_sum;
    uint_vec clusters_number;
    std::size_t size;

    void Resize(std::size_t s) {
        cells_prefix_sum.resize(s);
        clusters_prefix_sum.resize(s);
        clusters_number.resize(s);
        size = s;
    }
};

struct ModulesBuffer {
    uint_buf cells_prefix_sum;
    uint_buf clusters_prefix_sum;
    uint_buf clusters_number;
    std::size_t size;

    ModulesBuffer()
        : size(0) {
        cells_prefix_sum    = uint_buf();
        clusters_prefix_sum = uint_buf();
        clusters_number     = uint_buf();
    }

    void Resize(std::size_t s, vecmem::memory_resource& mr,
                 vecmem::cuda::async_copy& copy) {
        size = s;
        cells_prefix_sum    = uint_buf(s, mr);
        clusters_prefix_sum = uint_buf(s, mr);
        clusters_number     = uint_buf(s, mr);
        copy.setup(cells_prefix_sum);
        copy.setup(clusters_prefix_sum);
        copy.setup(clusters_number);
    }

    void CopyToDevice(const ModulesHost &modulesHost,
                      vecmem::cuda::async_copy& copy) {
        copy(vecmem::get_data(modulesHost.cells_prefix_sum),
                   cells_prefix_sum,
                   vecmem::copy::type::copy_type::host_to_device);
        // m_clusters_prefix_sum is filled in GPU
    }
};

struct ModulesView {
    uint_view cells_prefix_sum;
    uint_view clusters_prefix_sum;
    uint_view clusters_number;
    std::size_t size;

    ModulesView() = delete;
    ModulesView(const ModulesBuffer &m) {
        cells_prefix_sum    = m.cells_prefix_sum;
        clusters_prefix_sum = m.clusters_prefix_sum;
        clusters_number     = m.clusters_number;
        size = m.size;
    }
};

struct ModulesRefDevice {
    uint_device cells_prefix_sum;
    uint_device clusters_prefix_sum;
};

/*struct ClustersHost {
    uint_vec cells_prefix_sum;
    uint_vec module_id;
    std::size_t size;
};*/

struct ClustersBuffer {
    uint_buf cells_prefix_sum;
    uint_buf module_id;
    std::size_t size;

    ClustersBuffer()
        : size(0) {
        cells_prefix_sum    = uint_buf();
        module_id = uint_buf();
    }

    void Resize(std::size_t s, vecmem::memory_resource& mr,
                 vecmem::cuda::async_copy& copy) {
        size = s;
        cells_prefix_sum    = uint_buf(s, mr);
        module_id = uint_buf(s, mr);
        copy.setup(cells_prefix_sum);
        copy.setup(module_id);
    }

    /*void CopyToDevice(const ClustersHost &m,
                      vecmem::cuda::async_copy& copy) {
        copy(vecmem::get_data(m.cells_prefix_sum),
                   cells_prefix_sum,
                   vecmem::copy::type::copy_type::host_to_device);
        // m_clusters_prefix_sum is filled in GPU
    }*/
};

struct ClustersView {
    uint_view cells_prefix_sum;
    uint_view module_id;
    std::size_t size;

    ClustersView() = delete;
    ClustersView(const ClustersBuffer &c) {
        cells_prefix_sum    = c.cells_prefix_sum;
        module_id = c.module_id;
        size = c.size;
    }
};

struct ClustersRefDevice {
    uint_device cells_prefix_sum;
    uint_device module_id;
};

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
