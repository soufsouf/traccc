/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once
namespace traccc::device {
/*
 * Check if two cells are considered close enough to be part of the same
 * cluster.
 */
TRACCC_DEVICE
bool is_adjacent(channel_id ac0, channel_id ac1, channel_id bc0,
                 channel_id bc1) {
    unsigned int p0 = (ac0 - bc0);
    unsigned int p1 = (ac1 - bc1);
    return p0 * p0 <= 1 && p1 * p1 <= 1;
}
TRACCC_DEVICE
inline void reduce_problem_cell2(
    const unsigned short cid, const unsigned int start, const unsigned int end,
    unsigned char& adjc, unsigned short* adjv_part, 
    channel_id* channel0,channel_id* channel1,link_type* module_link,unsigned short* id_clusters) {
    //const unsigned int pos = cid + start;

    // Check if this code can benefit from changing to structs of arrays, as the
    // recurring accesses to cell data in global memory is slow right now.
    const channel_id c0 = channel0[cid];
    const channel_id c1 = channel1[cid];
    const unsigned int mod_id = module_link[cid];
    unsigned short min_id = cid;
    /*
     * First, we traverse the cells backwards, starting from the current
     * cell and working back to the first, collecting adjacent cells
     * along the way.
     */
    adjc =0;
    #pragma unroll
    for (unsigned short j = cid - 1; j < cid ; --j) {
        /*
         * Since the data is sorted, we can assume that if we see a cell
         * sufficiently far away in both directions, it becomes
         * impossible for that cell to ever be adjacent to this one.
         * This is a small optimisation.
         */
        
        if (channel1[j] + 1 < c1 || module_link[j] != mod_id) {
            break;
        }
        /*
         * If the cell examined is adjacent to the current cell, save it
         * in the current cell's adjacency set.
         */
        if (is_adjacent(c0, c1, channel0[j], channel1[j])) {
            adjv_part[adjc] = j ; 
            adjc ++;
            if((j-start)< min_id) min_id = j;
        }
    }
    /*
     * Now we examine all the cells past the current one, using almost
     * the same logic as in the backwards pass.
     */
    #pragma unroll
    for (unsigned int j = cid + 1; j + start < end; ++j) {
        /*
         * Note that this check now looks in the opposite direction! An
         * important difference.
         */
        if (channel1[j] > c1 + 1 || module_link[j] != mod_id) {
            break;
        }
        if (is_adjacent(c0, c1, channel0[j], channel1[j])) {
            adjv_part[adjc] = j; 
            adjc ++;
            if((j)< min_id) min_id = j;
        }
    }
    id_clusters[cid]= min_id;
    
    // if(blockIdx.x == 60 ||blockIdx.x == 100 || blockIdx.x == 10 )
    //printf(" blockIdx.x : %u | cid: %u |  id_fathers[cid]= %hu \n",blockIdx.x,cid, id_fathers[cid]);
}
}  // namespace traccc::device