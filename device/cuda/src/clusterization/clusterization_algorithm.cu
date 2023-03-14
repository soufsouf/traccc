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
#include "traccc/clusterization/device/aggregate_cluster.hpp"
#include "traccc/clusterization/device/form_spacepoints.hpp"
#include "traccc/clusterization/device/reduce_problem_cell.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <algorithm>

namespace traccc::cuda {

namespace {
/// These indices in clusterization will only range from 0 to
/// max_cells_per_partition, so we only need a short.
using index_t = unsigned short;

static constexpr int TARGET_CELLS_PER_THREAD = 8;
static constexpr int MAX_CELLS_PER_THREAD = 12;
}  // namespace

namespace kernels {

/// Implementation of a FastSV algorithm with the following steps:
///   1) mix of stochastic and aggressive hooking
///   2) shortcutting
///
/// The implementation corresponds to an adapted versiion of Algorithm 3 of
/// the following paper:
/// https://www.sciencedirect.com/science/article/pii/S0743731520302689
///
/// @param[inout] f     array holding the parent cell ID for the current
/// iteration.
/// @param[inout] gf    array holding grandparent cell ID from the previous
/// iteration.
///                     This array only gets updated at the end of the iteration
///                     to prevent race conditions.
/// @param[in] adjc     The number of adjacent cells
/// @param[in] adjv     Vector of adjacent cells
/// @param[in] tid      The thread index
///
__device__ void fast_sv_1(index_t* f, index_t* gf,
                          unsigned char adjc[MAX_CELLS_PER_THREAD],
                          index_t adjv[MAX_CELLS_PER_THREAD][8], index_t tid,
                          const index_t blckDim) {
    /*
     * The algorithm finishes if an iteration leaves the arrays unchanged.
     * This varible will be set if a change is made, and dictates if another
     * loop is necessary.
     */
    bool gf_changed;
   
    do {
        /*
         * Reset the end-parameter to false, so we can set it to true if we
         * make a change to the gf array.
         */
        gf_changed = false;

        /*
         * The algorithm executes in a loop of three distinct parallel
         * stages. In this first one, a mix of stochastic and aggressive
         * hooking, we examine adjacent cells and copy their grand parents
         * cluster ID if it is lower than ours, essentially merging the two
         * together.
         */
        for (index_t tst = 0; tst < MAX_CELLS_PER_THREAD; ++tst) {
            const index_t cid = tst * blckDim + tid;

            __builtin_assume(adjc[tst] <= 8);
            for (unsigned char k = 0; k < adjc[tst]; ++k) {
                index_t q = gf[adjv[tst][k]];

                if (gf[cid] > q) {
                    f[f[cid]] = q;
                    f[cid] = q;
                }
            }
           
             
        }
        
        
        /*
         * Each stage in this algorithm must be preceded by a
         * synchronization barrier!
         */
        __syncthreads();

#pragma unroll
        for (index_t tst = 0; tst < MAX_CELLS_PER_THREAD; ++tst) {
            const index_t cid = tst * blckDim + tid;
            /*
             * The second stage is shortcutting, which is an optimisation that
             * allows us to look at any shortcuts in the cluster IDs that we
             * can merge without adjacency information.
             */
            if (f[cid] > gf[cid]) {
                f[cid] = gf[cid];
            }
        }

        /*
         * Synchronize before the final stage.
         */
        __syncthreads();

#pragma unroll
        for (index_t tst = 0; tst < MAX_CELLS_PER_THREAD; ++tst) {
            const index_t cid = tst * blckDim + tid;
            /*
             * Update the array for the next generation, keeping track of any
             * changes we make.
             */
            if (gf[cid] != f[f[cid]]) {
                gf[cid] = f[f[cid]];
                gf_changed = true;
            }
        }

        /*
         * To determine whether we need another iteration, we use block
         * voting mechanics. Each thread checks if it has made any changes
         * to the arrays, and votes. If any thread votes true, all threads
         * will return a true value and go to the next iteration. Only if
         * all threads return false will the loop exit.
         */
    } while (__syncthreads_or(gf_changed));
}

__global__ void ccl_kernel(
    const alt_cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    const unsigned short max_cells_per_partition,
    const unsigned short target_cells_per_partition,
    spacepoint_collection_types::view spacepoints_view,
    unsigned int& measurement_count,
    vecmem::data::vector_view<unsigned int> cell_links) {

    const index_t tid = threadIdx.x;
    const index_t blckDim = blockDim.x;

    const alt_cell_collection_types::const_device cells_device(cells_view);
    spacepoint_collection_types::device spacepoints_device(spacepoints_view);
    const unsigned int num_cells = cells_device.size();
    __shared__ unsigned int start, end;
    /*
     * This variable will be used to write to the output later.
     */
    __shared__ unsigned int outi;
    extern __shared__ cluster fathers[];
    cluster* id_fathers = &fathers[0];
   // index_t* f = &fathers[max_cells_per_partition];
    //index_t* f_next = &fathers[2*max_cells_per_partition];
    /*
     * First, we determine the exact range of cells that is to be examined by
     * this block of threads. We start from an initial range determined by the
     * block index multiplied by the target number of cells per block. We then
     * shift both the start and the end of the block forward (to a later point
     * in the array); start and end may be moved different amounts.
     */
    if (tid == 0) {
        /*
         * Initialize shared variables.
         */
        start = blockIdx.x * target_cells_per_partition;
        assert(start < num_cells);
        end = std::min(num_cells, start + target_cells_per_partition);
        outi = 0;

        /*
         * Next, shift the starting point to a position further in the array;
         * the purpose of this is to ensure that we are not operating on any
         * cells that have been claimed by the previous block (if any).
         */
        while (start != 0 &&
               cells_device[start - 1].module_link ==
                   cells_device[start].module_link &&
               cells_device[start].c.channel1 <=
                   cells_device[start - 1].c.channel1 + 1) {
            ++start;
        }

        /*
         * Then, claim as many cells as we need past the naive end of the
         * current block to ensure that we do not end our partition on a cell
         * that is not a possible boundary!
         */
        while (end < num_cells &&
               cells_device[end - 1].module_link ==
                   cells_device[end].module_link &&
               cells_device[end].c.channel1 <=
                   cells_device[end - 1].c.channel1 + 1) {
            ++end;
        }
    }
    __syncthreads();

    const index_t size = end - start;
    assert(size <= max_cells_per_partition);

    // Check if any work needs to be done
    if (tid >= size) {
        return;
    }

    const cell_module_collection_types::const_device modules_device(
        modules_view);

  
    index_t adjv[MAX_CELLS_PER_THREAD*8];
   
    unsigned char adjc[MAX_CELLS_PER_THREAD];

#pragma unroll
     for (index_t tst = 0, cid; (cid = tst * blckDim + tid) < size; ++tst) {
        //adjc[tst] = 0;
        id_fathers[cid].channel0 = cells_device[cid+start].c.channel0;
        id_fathers[cid].channel1 = cells_device[cid+start].c.channel1;
        id_fathers[cid].activation = cells_device[cid+start].c.activation;
         id_fathers[cid].module_link = cells_device[cid+start].module_link;

    }
   
    __syncthreads();

    //unsigned short old_id,new_id;
    
bool gf_changed;
#pragma unroll
    for (index_t tst = 0, cid; (cid = tst * blckDim + tid) < size; ++tst) {
        /*
         * Look for adjacent cells to the current one.
         */
        device::reduce_problem_cell2(cid, start, end, adjc[tst],
                                    adjv[tst * MAX_CELLS_PER_THREAD],id_fathers);
      
       
    }
    
 __syncthreads();
   
      do {
        
        gf_changed = false;
              ///the father is the cell that has no small neighbors
              for (index_t tst = 0, cid; (cid = tst * blckDim + tid) < size; tst ++) {
               // if my father is not a real father then i have to communicate with neighbors  tothe find the real fahter

                for (index_t i = 0; i < adjc[tst]; i ++){    // neighbors communication
                if (id_fathers[cid].id_cluster > id_fathers[adjv[tst][i]].id_cluster) 
                {
                    id_fathers[cid].id_cluster = id_fathers[adjv[tst][i]].id_cluster;
                    gf_changed = true; 
                }
                
                }

       } 
    }while (__syncthreads_or(gf_changed));
    
            
    //printf("hello \n");
    
__syncthreads();

    for (index_t tst = 0, cid; (cid = tst * blckDim + tid) < size; ++tst) {
       // printf("f : %hu | id_fathers : %hu\n", f[cid],id_fathers[cid]);
        if (id_fathers[cid].id_cluster == cid) {
            atomicAdd(&outi, 1);
        }
    }

    __syncthreads();

    if (tid == 0) {
        outi = atomicAdd(&measurement_count, outi);
    }

    __syncthreads();

    const unsigned int groupPos = outi;

    __syncthreads();

    if (tid == 0) {
        outi = 0;
    }

    __syncthreads();

 

    for (index_t tst = 0, cid; (cid = tst * blckDim + tid) < size; ++tst) {
        if (id_fathers[cid].id_cluster == cid) {
            const unsigned int id = atomicAdd(&outi, 1);
            device::aggregate_cluster(
                 modules_device, id_fathers, start, end, cid,
                spacepoints_device, cell_links, groupPos + id);
        }
    }
}

__global__ void form_spacepoints(
    alt_measurement_collection_types::const_view measurements_view,
    cell_module_collection_types::const_view modules_view,
    const unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view) {

    device::form_spacepoints(threadIdx.x + blockIdx.x * blockDim.x,
                             measurements_view, modules_view, measurement_count,
                             spacepoints_view);
}

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    const unsigned short target_cells_per_partition)
    : m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_target_cells_per_partition(target_cells_per_partition) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const alt_cell_collection_types::const_view& cells,
    const cell_module_collection_types::const_view& modules) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Number of cells
    const alt_cell_collection_types::view::size_type num_cells =
        m_copy.get_size(cells);

    // Create result object for the CCL kernel with size overestimation
   /** alt_measurement_collection_types::buffer measurements_buffer(num_cells,
                                                                 m_mr.main);*/

    // Counter for number of measurements
    spacepoint_collection_types::buffer spacepoints_buffer(
        0.3*num_cells, m_mr.main);
   vecmem::unique_alloc_ptr<unsigned int> num_measurements_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    CUDA_ERROR_CHECK(cudaMemsetAsync(num_measurements_device.get(), 0,
                                     sizeof(unsigned int), stream));
    
    const unsigned short max_cells_per_partition =
        (m_target_cells_per_partition * MAX_CELLS_PER_THREAD +
         TARGET_CELLS_PER_THREAD - 1) /
        TARGET_CELLS_PER_THREAD;
    const unsigned int threads_per_partition =
        (m_target_cells_per_partition + TARGET_CELLS_PER_THREAD - 1) /
        TARGET_CELLS_PER_THREAD;
    const unsigned int num_partitions =
        (num_cells + m_target_cells_per_partition - 1) /
        m_target_cells_per_partition;

    // Create buffer for linking cells to their spacepoints.
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);

    // Launch ccl kernel. Each thread will handle a single cell.
    kernels::
        ccl_kernel<<<num_partitions, threads_per_partition,
                      max_cells_per_partition * sizeof(traccc::cluster), stream>>>(
            cells, modules, max_cells_per_partition,
            m_target_cells_per_partition, spacepoints_buffer,
            *num_measurements_device, cell_links);

    CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy number of measurements to host
    /*vecmem::unique_alloc_ptr<unsigned int> num_measurements_host =
        vecmem::make_unique_alloc<unsigned int>(*(m_mr.host));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        num_measurements_host.get(), num_measurements_device.get(),
        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    m_stream.synchronize();

    spacepoint_collection_types::buffer spacepoints_buffer(
        num_cells/2, m_mr.main);

    // For the following kernel, we can now use whatever the desired number of
    // threads per block.
    auto spacepointsLocalSize = 1024;
    const unsigned int num_blocks =
        (*num_measurements_host + spacepointsLocalSize - 1) /
        spacepointsLocalSize;

    // Turn 2D measurements into 3D spacepoints
    kernels::form_spacepoints<<<num_blocks, spacepointsLocalSize, 0, stream>>>(
        measurements_buffer, modules, *num_measurements_host,
        spacepoints_buffer);

    CUDA_ERROR_CHECK(cudaGetLastError());*/
    m_stream.synchronize();

    return {std::move(spacepoints_buffer), std::move(cell_links)};
}

}  // namespace traccc::cuda