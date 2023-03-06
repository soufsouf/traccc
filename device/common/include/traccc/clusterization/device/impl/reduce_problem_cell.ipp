/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <list>

#pragma once

namespace traccc::device {

/*
 * Check if two cells are considered close enough to be part of the same
 * cluster.
 */
using index_t = unsigned short;
TRACCC_HOST_DEVICE
bool is_adjacent(channel_id ac0, channel_id ac1, channel_id bc0,
                 channel_id bc1) {
    unsigned int p0 = (ac0 - bc0);
    unsigned int p1 = (ac1 - bc1);

    return p0 * p0 <= 1 && p1 * p1 <= 1;
}

TRACCC_HOST_DEVICE
inline void reduce_problem_cell(
    const alt_cell_collection_types::const_device& cells,
    const unsigned short cid, const unsigned int start, const unsigned int end, 
    grp_cluster* cluster_group,index_t cluster_count, idx_cluster* index) {

     const unsigned int pos = cid + start;
     //pos - 1= (tst * blckDim + tid )

     // Check if this code can benefit from changing to structs of arrays, as the
     // recurring accesses to cell data in global memory is slow right now.
     const channel_id c0 = cells[pos].c.channel0;
     const channel_id c1 = cells[pos].c.channel1;
     const unsigned int mod_id = cells[pos].module_link;

     /*
     * First, we traverse the cells backwards, starting from the current
     * cell and working back to the first, collecting adjacent cells
     * along the way.
     */
     unsigned int i = pos - 1; 
     bool find = false;
     unsigned int& idx = index[cid].id_cluster;

     unsigned short write_done;
     while (cells[i].c.channel1 + 1 >= c1 && cells[i].module_link == mod_id  && i > (start - 1))
       {
         if (is_adjacent(c0, c1, cells[i].c.channel0, cells[i].c.channel1)) {
          while (!index[i - start].write) 
          {
          write_done = 0;
          }
         __threadfence();

          unsigned int idx_cells = index[i - start].id_cluster + 1;
          index[pos - start].module_link= mod_id;
          atomicExch(&index[pos - start].id_cluster, idx_cells );
          __threadfence();
          cluster_group[idx_cells].cluster_link = pos;
          cluster_group[idx_cells].write = 1 ;
          atomicExch(&index[pos - start].write, 1); 
          find = true;
         break;
            }
            i -- ;
        }
        
    
    if ( find ==false)
    {   index[pos - start].module_link = mod_id;
        atomicAdd(&cluster_count, 1);
       atomicExch(&index[pos - start].id_cluster,cluster_count );
       cluster_group[cluster_count*8].cluster_link= pos;
       cluster_group[cluster_count*8].write = 1 ;
       __threadfence();
       atomicExch(&index[pos - start].write, 1); 
       
    }

    

    
}

}  // namespace traccc::device