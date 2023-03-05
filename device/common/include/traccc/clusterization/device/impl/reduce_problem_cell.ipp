/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#include <unordered_map>
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
    unsigned char& adjc, unsigned short adjv[8], 
    std::unordered_map<index_t, std::list<index_t>>* cluster_map) {

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
    while (cells[i].c.channel1 + 1 >= c1 && cells[i].module_link == mod_id  && i > (start - 1))
    {
        if (is_adjacent(c0, c1, cells[i].c.channel0, cells[i].c.channel1)) {
            uint64_t idx = cells[i].c.cluster_indice;
            if( idx != 2000)
            {
                cells[pos].c.cluster_indice = idx;
                std::list<index_t>& values = cluster_map[idx];
                values.push_back(pos);
                find = true;
                break;
            }
        }
    }
    if ( find ==false)
    {
       
       (*cluster_map).insert(std::pair(cluster_map.size(), std::list<index_t>()));
       std::list<index_t>& new_pair = cluster_map[(*cluster_map).size()];
       new_pair.push_back(pos);
       cells[pos].c.cluster_indice = (*cluster_map).size();
    }

    for (unsigned int j = pos - 1; j < pos; --j) {
        /*
         * Since the data is sorted, we can assume that if we see a cell
         * sufficiently far away in both directions, it becomes
         * impossible for that cell to ever be adjacent to this one.
         * This is a small optimisation.
         */
        if (cells[j].c.channel1 + 1 < c1 || cells[j].module_link != mod_id) {
            break;
        }

        /*
         * If the cell examined is adjacent to the current cell, save it
         * in the current cell's adjacency set.
         */
        if (is_adjacent(c0, c1, cells[j].c.channel0, cells[j].c.channel1)) {
            adjv[adjc++] = j - start;
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
            adjv[adjc++] = j - start;
        }
    }
}

}  // namespace traccc::device