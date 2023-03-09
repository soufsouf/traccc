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
TRACCC_DEVICE
bool is_adjacent(channel_id ac0, channel_id ac1, channel_id bc0,
                 channel_id bc1) {
    unsigned int p0 = (ac0 - bc0);
    unsigned int p1 = (ac1 - bc1);

    return p0 * p0 <= 1 && p1 * p1 <= 1;
}

TRACCC_DEVICE
inline void reduce_problem_cell(
    const alt_cell_collection_types::const_device& cells,
    const unsigned short cid, const unsigned int start, const unsigned int end, 
    index_t* cluster_group,unsigned int *cluster_count) {

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
     
     int cluster_id = 0;
     int j = (cid == 0? start: pos - 1);
     int counter = 0 ;
     if(cid == 0)
     {
      //printf(" hello reduce cell 1 \n"); 
        cluster_group[cid]= cid;
        

     }
     else
     { 
      //printf(" hello reduce cell 2 \n"); 
        while( j > start && (cells[j].c.channel1 + 1 )<= c1 && (cells[j].module_link == mod_id))
        {
          //printf(" hello reduce cell 3 \n"); 
          if (is_adjacent(c0, c1, cells[j].c.channel0, cells[j].c.channel1)) 
          {
           // printf(" hello reduce cell 4 \n"); 
            cluster_id = j - start;
            
            counter ++;
          }
          j --;
        }
        //printf(" hello reduce cell 5 \n"); 
          cluster_group[cid] = cluster_id ;
          
      }
     //printf(" hello reduce cell 6 \n"); 
     if(counter == 0)
     atomicAdd(cluster_count,1);
     //printf(" hello reduce cell 7 \n"); 
}

}  // namespace traccc::device