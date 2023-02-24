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
#include "traccc/edm/cell.hpp"
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


using scalar = TRACCC_CUSTOM_SCALARTYPE;
namespace traccc::cuda {
    
namespace kernels {

__global__ void find_clusters(
    const CellsView cellsView,
    const ModulesView modulesView,
    vecmem::data::vector_view<unsigned int> clusters_per_module_view) {

    device::find_clusters(threadIdx.x + blockIdx.x * blockDim.x,
                          cellsView, modulesView,
                          clusters_per_module_view);
}

__global__ void count_cluster_cells(
    vecmem::data::vector_view<unsigned int> cluster_prefix_sum_view,
    const CellsView cellsView,
    vecmem::data::vector_view<unsigned int> clusters_view) {

    device::count_cluster_cells(
        threadIdx.x + blockIdx.x * blockDim.x,
        cluster_prefix_sum_view , cellsView , clusters_view );
}

__global__ void connect_components(
     const CellsView cellsView,
     vecmem::data::vector_view<unsigned int> cluster_prefix_sum_view,
     cluster_container_types::view  clusters_view) {

    device::connect_components(threadIdx.x + blockIdx.x * blockDim.x,
                               cellsView,
                               cluster_prefix_sum_view,
                               clusters_view);
}

__global__ void create_measurements(
    cluster_container_types::const_view clusters_view,
    const traccc::HeadersView headersView,
    vecmem::data::vector_view<unsigned int > Clusters_module_link ,
    vecmem::data::vector_view<point2 > measurement_local,
    vecmem::data::vector_view<point2 > measurement_variance) {

    device::create_measurements(threadIdx.x + blockIdx.x * blockDim.x,
                                clusters_view, headersView,
                                Clusters_module_link, measurement_local,
                                measurement_variance);
}

__global__ void form_spacepoints(
    const traccc::HeadersView headersView,
    vecmem::data::vector_view<unsigned int > Clusters_module_link ,
    vecmem::data::vector_view<point2 > measurement_local,
    vecmem::data::vector_view<point2 > measurement_variance,
    spacepoint_container_types::view spacepoints_view) {
        device::form_spacepoints(threadIdx.x + blockIdx.x * blockDim.x,
                    headersView,Clusters_module_link,
     measurement_local , measurement_variance , spacepoints_view );
        
        
}

}  // namespace kernels

clusterization_algorithm2::clusterization_algorithm2(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str)
    : m_mr(mr), m_copy(copy), m_stream(str) {}

clusterization_algorithm2::output_type clusterization_algorithm2::operator()(
    const traccc::CellsView& cellsView, const traccc::ModulesView& modulesView,
    const traccc::HeadersView& headersView) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Number of modules
    unsigned int num_modules = modulesView.size;

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
    
    vecmem::data::vector_buffer<unsigned int> cl_per_module_prefix_buff(num_modules, m_mr.main);
    m_copy.setup(cl_per_module_prefix_buff);

    // Work block size for kernel execution
    const std::size_t threadsPerBlock = 64;
    // Calculating grid size for cluster finding kernel
    std::size_t blocksPerGrid =
        (num_modules + threadsPerBlock - 1) / threadsPerBlock;

    // Invoke find clusters that will call cluster finding kernel
    kernels::find_clusters<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        cellsView, modulesView, cl_per_module_prefix_buff);
    CUDA_ERROR_CHECK(cudaGetLastError());

    vecmem::vector<unsigned int> cl_per_module_prefix_host(
        m_mr.host ? m_mr.host : &(m_mr.main));
         
    m_copy(cl_per_module_prefix_buff, cl_per_module_prefix_host,
           vecmem::copy::type::copy_type::device_to_host);
       
    m_stream.synchronize();

    //for(int i = 23; i<29;i++) printf("label 1 : %u \n",cl_per_module_prefix_host[i] );    
    std::vector<std::size_t> clusters_per_module_host(
        cl_per_module_prefix_host.begin(), cl_per_module_prefix_host.end());
  //for(int j = 5; j<20;j++) printf("host avant IS : %llu *** \n",cl_per_module_prefix_host[j] );
    
    // Perform the inclusive scan operation
    std::inclusive_scan(cl_per_module_prefix_host.begin(),
                        cl_per_module_prefix_host.end(),
                        cl_per_module_prefix_host.begin());

    unsigned int total_clusters = cl_per_module_prefix_host.back();
   /* printf(" n cluster %u \n", total_clusters );
    for(int i = 230; i<290;i++) printf("label 2 : %u \n",cl_per_module_prefix_host[i] );*/
//for(int i = 230; i<261;i++) printf("host : %llu \n",cl_per_module_prefix_host[i] );
    // Copy the prefix sum back to its device container
    m_copy(vecmem::get_data(cl_per_module_prefix_host),
           modulesView.clusters_prefix_sum,
           vecmem::copy::type::copy_type::host_to_device);

    // Vector of the exact cluster sizes, will be filled in cluster counting
    vecmem::data::vector_buffer<unsigned int> cluster_sizes_buffer(
        total_clusters, m_mr.main);
    m_copy.setup(cluster_sizes_buffer);
    m_copy.memset(cluster_sizes_buffer, 0);


    //printf("capacity : %llu\n", cells_prefix_sum_buff.capacity());
    // Calclating grid size for cluster counting kernel (block size 64)
    blocksPerGrid = (cellsView.size + threadsPerBlock - 1) /
                    threadsPerBlock;
    // Invoke cluster counting will call count cluster cells kernel
    vecmem::data::vector_buffer<unsigned int> cells_cluster_ps(total_clusters, m_mr.main);
    m_copy.setup(cells_cluster_ps);//prefix sum cells per cluster 

    kernels::count_cluster_cells<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        cl_per_module_prefix_buff, cellsView, cluster_sizes_buffer);
    // Check for kernel launch errors and Wait for the cluster_counting kernel
    // to finish
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy cluster sizes back to the host
    vecmem::vector<unsigned int> cluster_sizes{m_mr.host ? m_mr.host
                                                         : &(m_mr.main)};
    m_copy(cluster_sizes_buffer, cluster_sizes,
           vecmem::copy::type::copy_type::device_to_host);
   
    m_stream.synchronize();
 //for(int i = 20; i<30;i++) printf("size : %u \n",cluster_sizes[i] );
    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_types::buffer clusters_buffer{
        {total_clusters, m_mr.main},
        {std::vector<std::size_t>(total_clusters, 0),
         std::vector<std::size_t>(cluster_sizes.begin(), cluster_sizes.end()),
         m_mr.main, m_mr.host}};
    m_copy.setup(clusters_buffer.headers);
    m_copy.setup(clusters_buffer.items);

  /* vecmem::data::jagged_vector_buffer<unsigned int> clusters_buff(
        std::vector<unsigned int>(cluster_sizes.begin(), cluster_sizes.end()),
        m_mr.main, m_mr.host);
    m_copy.setup(clusters_buff); */

    // Using previous block size and thread size (64)
    // Invoke connect components will call connect components kernel
    kernels::connect_components<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        cellsView,
        cl_per_module_prefix_buff,
        clusters_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError()); 

    // Resizable buffer for the measurements
   

    // Spacepoint container buffer to fill inside the spacepoint formation
    // kernel
    spacepoint_container_types::buffer spacepoints_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy.setup(spacepoints_buffer.headers);
    m_copy.setup(spacepoints_buffer.items);

   // printf("clusters_buff %llu\n", clusters_buffer.headers.size());
    // Calculating grid size for measurements creation kernel (block size 64)
    blocksPerGrid = (clusters_buffer.headers.size() - 1 + threadsPerBlock) /
                    threadsPerBlock;
    
    
    //measurement struct 
    vecmem::data::vector_buffer<unsigned int> Clusters_module_link(total_clusters, m_mr.main);
    m_copy.setup(Clusters_module_link);
    m_copy.memset(Clusters_module_link, 0);

    vecmem::data::vector_buffer<point2> measurement_local(total_clusters, m_mr.main);
    m_copy.setup(measurement_local);
    m_copy.memset(measurement_local, 0);

    vecmem::data::vector_buffer<point2> measurement_variance(total_clusters, m_mr.main);
    m_copy.setup(measurement_variance);
    //m_copy.memset(measurement_variance, 0);
   /* vecmem::data::vector_buffer<point3> global(total_clusters, m_mr.main);
    m_copy.setup(global);
    m_copy.memset(global, 0); */


    // Invoke measurements creation will call create measurements kernel
    kernels::create_measurements<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        clusters_buffer, headersView, Clusters_module_link, measurement_local,
        measurement_variance);
    CUDA_ERROR_CHECK(cudaGetLastError());
   
   
    // Using the same grid size as before
    // Invoke spacepoint formation will call form_spacepoints kernel

    kernels::form_spacepoints<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
       headersView, Clusters_module_link,measurement_local,measurement_variance , spacepoints_buffer );
    CUDA_ERROR_CHECK(cudaGetLastError());
    
    // Return the buffer. Which may very well not be filled at this point yet.
    return spacepoints_buffer;
}




clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str)
    : m_mr(mr), m_copy(copy), m_stream(str) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_container_types::const_view& cells_view ) const {

    spacepoint_container_types::buffer spacepoints_buffer{
        {50, m_mr.main},
        {std::vector<std::size_t>(50, 0), std::vector<std::size_t>(50, 0),
         m_mr.main, m_mr.host}};
    return spacepoints_buffer;
}

}  // namespace traccc::cuda


