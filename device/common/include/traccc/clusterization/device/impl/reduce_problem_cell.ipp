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
    const alt_cell_collection_types::const_device& cells,
    const unsigned short cid, const unsigned int start, const unsigned int end,
    unsigned char& adjc, unsigned short adjv[8],unsigned short* id_fathers) {

    const unsigned int pos = cid + start;

    // Check if this code can benefit from changing to structs of arrays, as the
    // recurring accesses to cell data in global memory is slow right now.
    const channel_id c0 = cells[pos].c.channel0;
    const channel_id c1 = cells[pos].c.channel1;
    const unsigned int mod_id = cells[pos].module_link;
    unsigned short min_id = cid;
    unsigned short old_id,new_id;
    bool count = false;
    /*
     * First, we traverse the cells backwards, starting from the current
     * cell and working back to the first, collecting adjacent cells
     * along the way.
     */
    for (unsigned int j = pos - 1; j < pos; --j) {
        /*
         * Since the data is sorted, we can assume that if we see a cell
         * sufficiently far away in both directions, it becomes
         * impossible for that cell to ever be adjacent to this one.
         * This is a small optimisation.
         */
        assert(j >= start);
        if (cells[j].c.channel1 + 1 < c1 || cells[j].module_link != mod_id) {
            break;
        }

        /*
         * If the cell examined is adjacent to the current cell, save it
         * in the current cell's adjacency set.
         */
        if (is_adjacent(c0, c1, cells[j].c.channel0, cells[j].c.channel1)) {
            adjv[adjc] = j - start; 
            adjc ++;
            if((j-start)< min_id) min_id = j-start;
        }
    }

    /*
     * Now we examine all the cells past the current one, using almost
     * the same logic as in the backwards pass.
     */
    for (unsigned int j = pos + 1; j < end; ++j) {
        /*
         * Note that this check now looks in the opposite direction! An
         * important difference.
         */
        if (cells[j].c.channel1 > c1 + 1 || cells[j].module_link != mod_id) {
            break;
        }

        if (is_adjacent(c0, c1, cells[j].c.channel0, cells[j].c.channel1)) {
            adjv[adjc] = j - start; 
            adjc ++;
            if((j-start)< min_id) min_id = j-start;
        }
    }
    id_fathers[cid]= min_id;
    printf(" id_fathers[cid]= %hu \n", id_fathers[cid]);
    __syncthreads();
    do{
        count = false; 
        old_id = id_fathers[adjv[cid]];
        #pragma unroll
      for(unsigned char i = 0; i< adjc; i ++)
      {
       
        if(id_fathers[adjv[i]] < old_id)
               {new_id = id_fathers[adjv[i]];
               count =true; }
               
      }
      id_fathers[adjv[cid]] = new_id;
     printf("hello 2\n");
    }while(__syncthreads_or(count));
//printf("hello \n");

}



}  // namespace traccc::device