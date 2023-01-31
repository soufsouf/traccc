/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../utils/utils.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s)
#include "traccc/clusterization/device/connect_components.hpp"
#include "traccc/clusterization/device/count_cluster_cells.hpp"
#include "traccc/clusterization/device/create_measurements.hpp"
#include "traccc/clusterization/device/find_clusters.hpp"
#include "traccc/clusterization/device/form_spacepoints.hpp"
#include "traccc/cuda/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/fill_prefix_sum.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <algorithm>

std::size_t cellcount;
using scalar = TRACCC_CUSTOM_SCALARTYPE;
namespace traccc::cuda {
namespace kernels {

__global__ void fill_buffers(const cell_container_types::const_view cells_view,
                             vecmem::data::vector_view<unsigned int> channel0,
                             vecmem::data::vector_view<unsigned int> channel1,
vecmem::data::vector_view<scalar> activat,
                             vecmem::data::vector_view<unsigned int> cumulsize,
                             vecmem::data::vector_view<unsigned int> moduleidx) {

    cell_container_types::const_device cells_device(cells_view);
    vecmem::device_vector<unsigned int> ch0(channel0);
    vecmem::device_vector<unsigned int> ch1(channel1);
    vecmem::device_vector<scalar> activation(activat);
    vecmem::device_vector<unsigned int> sum(cumulsize);
    vecmem::device_vector<unsigned int> midx(moduleidx);

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= cells_device.size())
        return;

    const auto& cells = cells_device.at(idx).items;
    
    unsigned int doffset = 0;
    for (int i=0; i < idx; i++) {
        doffset+= cells_device.at(i).items.size();
    }
    sum.at(idx) = doffset;

    if (idx == cells_device.size() - 1) {
        sum.at(idx+1) = doffset + cells_device.at(idx).items.size();
    }

    std::size_t n_cells = cells.size();
    for (int i=0; i < n_cells; i++) {
        ch0.at(i+doffset) = cells[i].channel0;
        ch1.at(i+doffset) = cells[i].channel1;
activation.at(i+doffset)=cells[i].activation;
        midx.at(i+doffset) = idx;
    }
}

__global__ void find_clusters(
    const cell_container_types::const_view cells_view,
    vecmem::data::vector_view<unsigned int> channel0,
    vecmem::data::vector_view<unsigned int> channel1,
    vecmem::data::vector_view<unsigned int> cumulsize,
    vecmem::data::vector_view<unsigned int> moduleidx,
    vecmem::data::vector_view<unsigned int> label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    device::find_clusters(threadIdx.x + blockIdx.x * blockDim.x, cells_view,
                          channel0, channel1, cumulsize, moduleidx,
                          label_view, clusters_per_module_view);
}

__global__ void fill2(vecmem::data::vector_view<unsigned int> label_view,
                      vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
                      vecmem::data::vector_view<unsigned int> cumulsize) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= cumulsize.size()-1)
        return;
    
    vecmem::device_vector<unsigned int> labels(label_view);
    vecmem::jagged_device_vector<unsigned int> device_sparse_ccl_indices(
        sparse_ccl_indices_view);
    vecmem::device_vector<unsigned int> sum(cumulsize);

    unsigned int doffset = sum[idx];
    const unsigned int n_cells = sum[idx+1] - doffset;
    for (int i=0; i < n_cells; i++) {
        device_sparse_ccl_indices[idx][i] = labels[i+doffset];
    }
}

 __global__ void fill3(const cell_container_types::const_view cells_view,
    vecmem::data::vector_view<unsigned int > Clusters_module_link ,
    vecmem::data::vector_view<point2 > measurement_local,
    vecmem::data::vector_view<variance2 > measurement_variance,
    measurement_container_types::view measurements_view )
    {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx >= Clusters_module_link.size())
         return;
    cell_container_types::const_device cells_device(cells_view);
    vecmem::device_vector<unsigned int> Cl_module_link(Clusters_module_link);
    vecmem::device_vector<point2> local_measurement(measurement_local);
    vecmem::device_vector<variance2> variance_measurement(measurement_variance);
    measurement_container_types::device measurements_device(measurements_view);
    
    std::size_t module_link_ = Cl_module_link[idx];
    point2 local_ = local_measurement[idx];
    variance2 variance_ = variance_measurement[idx];
    measurement m;
    m.cluster_link = module_link_;
    m.local = local_;
    m.variance = variance_;
    auto &module = cells_device.at(module_link_).header;
    measurements_device[module_link_].header = module;
    measurements_device[module_link_].items.push_back(std::move(m));
    }

__global__ void count_cluster_cells(
    vecmem::data::vector_view<unsigned int> label_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
     vecmem::data::vector_view<unsigned int> moduleidx,
   vecmem::data::vector_view<unsigned int> cells_cl_prefix_sum,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view) {

    device::count_cluster_cells(
        threadIdx.x + blockIdx.x * blockDim.x, label_view,
        cluster_prefix_sum_view,moduleidx, cells_cl_prefix_sum, cluster_sizes_view);
}

__global__ void connect_components(
     vecmem::data::vector_view<unsigned int> moduleidx,
     vecmem::data::vector_view<unsigned int> label_view,
     vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
     vecmem::data::vector_view<unsigned int> cluster_idx_atomic,
     vecmem::data::vector_view<unsigned int> cells_cl_prefix_sum,
    vecmem::data::vector_view<unsigned int> clusters_view) {

    device::connect_components(threadIdx.x + blockIdx.x * blockDim.x,
                               moduleidx, label_view,
                               cluster_prefix_sum_view, cluster_idx_atomic,cells_cl_prefix_sum,
                               clusters_view, 0);
}

__global__ void create_measurements(
    vecmem::data::vector_view<unsigned int > moduleidx,
    vecmem::data::vector_view<scalar> activation_cell,
    vecmem::data::vector_view<unsigned int> channel0,
    vecmem::data::vector_view<unsigned int> channel1,
    vecmem::data::vector_view<unsigned int > clusters_view,
    vecmem::data::vector_view<unsigned int > cel_cl_ps, // cell_cluster_prefix_sum
    const cell_container_types::const_view cells_view,
    measurement_container_types::view measurements_view ,
    vecmem::data::vector_view<unsigned int > Clusters_module_link ,
    vecmem::data::vector_view<point2 > measurement_local,
    vecmem::data::vector_view<variance2 > measurement_variance) {

    device::create_measurements(threadIdx.x + blockIdx.x * blockDim.x,
                              moduleidx ,activation_cell,channel0, channel1,
                                clusters_view,cel_cl_ps, cells_view,
                                 measurements_view,Clusters_module_link, measurement_local, measurement_variance);
}

__global__ void form_spacepoints(
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view,
    spacepoint_container_types::view spacepoints_view) {

    device::form_spacepoints(threadIdx.x + blockIdx.x * blockDim.x,
                             measurements_view, measurements_prefix_sum_view,
                             spacepoints_view);
}

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str)
    : m_mr(mr), m_copy(copy), m_stream(str) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_container_types::const_view& cells_view) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Number of modules
    const cell_container_types::const_device::header_vector::size_type
        num_modules = m_copy.get_size(cells_view.headers);

    // Work block size for kernel execution
    const std::size_t threadsPerBlock = 64;

    // Get the sizes of the cells in each module
    const std::vector<
        cell_container_types::const_device::item_vector::value_type::size_type>
        cell_sizes = m_copy.get_sizes(cells_view.items);

    cellcount = 0;
    for (int i=0; i < cell_sizes.size(); i++) {
        cellcount += cell_sizes[i];
    }

    //cellvec cells;
    vecmem::data::vector_buffer<unsigned int> channel0(cellcount, m_mr.main);
    m_copy.setup(channel0);
    vecmem::data::vector_buffer<unsigned int> channel1(cellcount, m_mr.main);
    m_copy.setup(channel1);
    vecmem::data::vector_buffer<scalar> activation(cellcount, m_mr.main);
    m_copy.setup(activation);
    vecmem::data::vector_buffer<unsigned int> moduleidx(cellcount, m_mr.main);
    m_copy.setup(moduleidx);
    vecmem::data::vector_buffer<unsigned int> prefixsum(num_modules+1, m_mr.main);
    m_copy.setup(prefixsum);

    std::size_t blocksPerGrid = (num_modules + threadsPerBlock - 1) / threadsPerBlock;
    kernels::fill_buffers<<<blocksPerGrid, threadsPerBlock, 0, stream>>>
                            (cells_view, channel0, channel1,activation, prefixsum, moduleidx);

    /*
     * Helper container for sparse CCL calculations.
     * Each inner vector corresponds to 1 module.
     * The indices in a particular inner vector will be filled by sparse ccl
     * and will indicate to which cluster, a particular cell in the module
     * belongs to.
     */
    /*vecmem::data::jagged_vector_buffer<unsigned int> sparse_ccl_indices_buff(
        std::vector<std::size_t>(cell_sizes.begin(), cell_sizes.end()),
        m_mr.main, m_mr.host);
    m_copy.setup(sparse_ccl_indices_buff);*/

    vecmem::data::vector_buffer<unsigned int> label_buff(cellcount, m_mr.main);
    m_copy.setup(label_buff);

    /*
     * cl_per_module_prefix_buff is a vector buffer with numbers of found
     * clusters in each module. Later it will be transformed into prefix sum
     * vector (hence the name). The logic is the following. After
     * cluster_finding_kernel, the buffer will contain cluster sizes e.i.
     *
     * cluster sizes: | 1 | 12 | 5 | 102 | 42 | ... - cl_per_module_prefix_buff
     * module index:  | 0 |  1 | 2 |  3  |  4 | ...
     *
     * Now, we copy those cluster sizes to the host and make a duplicate vector
     * of them. So, we are left with cl_per_module_prefix_host, and
     * clusters_per_module_host - which are the same. Now, we procede to
     * modifying the cl_per_module_prefix_host to actually resemble its name
     * i.e.
     *
     * We do std::inclusive_scan on it, which will result in a prefix sum
     * vector:
     *
     * cl_per_module_prefix_host: | 1 | 13 | 18 | 120 | 162 | ...
     *
     * Then, we copy this vector into the previous cl_per_module_prefix_buff.
     * In this way, we don't need to allocate the memory on the device twice.
     *
     * Now, the monotonic prefix sum buffer - cl_per_module_prefix_buff, will
     * allow us to insert the clusters at the correct position inside the
     * kernel. The remaining host vector - clusters_per_module_host, will be
     * needed to allocate memory for other buffers later in the code.
     */
    vecmem::data::vector_buffer<std::size_t> cl_per_module_prefix_buff(
        num_modules, m_mr.main);
    m_copy.setup(cl_per_module_prefix_buff);

    // Calculating grid size for cluster finding kernel
    blocksPerGrid =
        (num_modules + threadsPerBlock - 1) / threadsPerBlock;

    // Invoke find clusters that will call cluster finding kernel
    kernels::find_clusters<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        cells_view, channel0, channel1, prefixsum, moduleidx,
        label_buff, cl_per_module_prefix_buff);
    CUDA_ERROR_CHECK(cudaGetLastError());

    /*kernels::fill2<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        label_buff, sparse_ccl_indices_buff, prefixsum);
    CUDA_ERROR_CHECK(cudaGetLastError());*/

    // Create prefix sum buffer
    vecmem::data::vector_buffer cells_prefix_sum_buff =
        make_prefix_sum_buff(cell_sizes, m_copy, m_mr, m_stream);

    // Copy the sizes of clusters per module to the host
    // and create a copy of "clusters per module" vector
    vecmem::vector<std::size_t> cl_per_module_prefix_host(
        m_mr.host ? m_mr.host : &(m_mr.main));
    m_copy(cl_per_module_prefix_buff, cl_per_module_prefix_host,
           vecmem::copy::type::copy_type::device_to_host);
    m_stream.synchronize();
    std::vector<std::size_t> clusters_per_module_host(
        cl_per_module_prefix_host.begin(), cl_per_module_prefix_host.end());

    // Perform the inclusive scan operation
    std::inclusive_scan(cl_per_module_prefix_host.begin(),
                        cl_per_module_prefix_host.end(),
                        cl_per_module_prefix_host.begin());

    unsigned int total_clusters = cl_per_module_prefix_host.back();

    // Copy the prefix sum back to its device container
    m_copy(vecmem::get_data(cl_per_module_prefix_host),
           cl_per_module_prefix_buff,
           vecmem::copy::type::copy_type::host_to_device);

    // Vector of the exact cluster sizes, will be filled in cluster counting
    vecmem::data::vector_buffer<unsigned int> cluster_sizes_buffer(
        total_clusters, m_mr.main);
    m_copy.setup(cluster_sizes_buffer);
    m_copy.memset(cluster_sizes_buffer, 0);

    printf("capacity : %llu\n", cells_prefix_sum_buff.capacity());
    // Calclating grid size for cluster counting kernel (block size 64)
    blocksPerGrid = (cells_prefix_sum_buff.capacity() + threadsPerBlock - 1) /
                    threadsPerBlock;
    // Invoke cluster counting will call count cluster cells kernel
    vecmem::data::vector_buffer<unsigned int> cells_cluster_ps(total_clusters, m_mr.main);
    m_copy.setup(cells_cluster_ps);//prefix sum cells per cluster 
    kernels::count_cluster_cells<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        label_buff, cl_per_module_prefix_buff, moduleidx, cells_cluster_ps,
        cluster_sizes_buffer);
    // Check for kernel launch errors and Wait for the cluster_counting kernel
    // to finish
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy cluster sizes back to the host
    vecmem::vector<unsigned int> cluster_sizes{m_mr.host ? m_mr.host
                                                         : &(m_mr.main)};
    m_copy(cluster_sizes_buffer, cluster_sizes,
           vecmem::copy::type::copy_type::device_to_host);
    m_stream.synchronize();

    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_types::buffer clusters_buffer{
        {total_clusters, m_mr.main},
        {std::vector<std::size_t>(total_clusters, 0),
         std::vector<std::size_t>(cluster_sizes.begin(), cluster_sizes.end()),
         m_mr.main, m_mr.host}};
    m_copy.setup(clusters_buffer.headers);
    m_copy.setup(clusters_buffer.items);

    vecmem::data::vector_buffer<unsigned int> cluster_index_atomic(total_clusters, m_mr.main);
    m_copy.setup(cluster_index_atomic);
    m_copy.memset(cluster_index_atomic, 0);
    vecmem::data::vector_buffer<unsigned int> clusters_buff(cellcount, m_mr.main);
    m_copy.setup(clusters_buff);

    // Using previous block size and thread size (64)
    // Invoke connect components will call connect components kernel
    kernels::connect_components<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        moduleidx, label_buff, cl_per_module_prefix_buff, cluster_index_atomic,
        cells_cluster_ps, clusters_buff);
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Resizable buffer for the measurements
    measurement_container_types::buffer measurements_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy.setup(measurements_buffer.headers);
    m_copy.setup(measurements_buffer.items);

    // Spacepoint container buffer to fill inside the spacepoint formation
    // kernel
    spacepoint_container_types::buffer spacepoints_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy.setup(spacepoints_buffer.headers);
    m_copy.setup(spacepoints_buffer.items);

    printf("clusters_buff %llu\n", clusters_buffer.headers.size());
    // Calculating grid size for measurements creation kernel (block size 64)
    blocksPerGrid = (clusters_buffer.headers.size() - 1 + threadsPerBlock) /
                    threadsPerBlock;
    
    
    //measurement struct 
    vecmem::data::vector_buffer<unsigned int> Clusters_module_link(total_clusters, m_mr.main);
    m_copy.setup(Clusters_module_link);
    m_copy.memset(Clusters_module_link, 0);

    vecmem::data::vector_buffer<point2> measurement_local(total_clusters, m_mr.main);
    m_copy.setup(measurement_local);
    //m_copy.memset(measurement_local, 0);

    vecmem::data::vector_buffer<variance2> measurement_variance(total_clusters, m_mr.main);
    m_copy.setup(measurement_variance);
    //m_copy.memset(measurement_variance, 0);


    // Invoke measurements creation will call create measurements kernel
    kernels::create_measurements<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        moduleidx, activation ,channel0, channel1,clusters_buff,cells_cluster_ps,cells_view,
        measurements_buffer, Clusters_module_link,measurement_local, measurement_variance);
    CUDA_ERROR_CHECK(cudaGetLastError());
   
   
   //kernel fill 3 
   
    measurement_container_types::buffer measurement_buff{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy.setup(measurement_buff.headers);
    m_copy.setup(measurement_buff.items);

    kernels::fill3<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
       cells_view, Clusters_module_link,measurement_local, measurement_variance,measurement_buff );
    // Create prefix sum buffer
    vecmem::data::vector_buffer meas_prefix_sum_buff = make_prefix_sum_buff(
        std::vector<device::prefix_sum_size_t>{clusters_per_module_host.begin(),
                                               clusters_per_module_host.end()},
        m_copy, m_mr, m_stream);

    // Using the same grid size as before
    // Invoke spacepoint formation will call form_spacepoints kernel
    kernels::form_spacepoints<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        measurement_buff, meas_prefix_sum_buff, spacepoints_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Return the buffer. Which may very well not be filled at this point yet.
    return spacepoints_buffer;
}

}  // namespace traccc::cuda
