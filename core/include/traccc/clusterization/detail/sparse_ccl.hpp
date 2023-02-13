/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

// System include(s).
#include <cassert>

namespace traccc {

/// Implemementation of SparseCCL, following
/// [DOI: 10.1109/DASIP48288.2019.9049184]
///
/// Requires cells to be sorted in column major
namespace detail {

/// Find root of the tree for entry @param e
///
/// @param L an equivalance table
///
/// @return the root of @param e
template <typename ccl_vector_t>
TRACCC_HOST_DEVICE inline unsigned int find_root(const ccl_vector_t& L,
                                                 unsigned int e) {

    unsigned int r = e;
    assert(r < L.size());
    while (L[r] != r) {
        r = L[r];
        assert(r < L.size());
    }
    return r;
}

template <typename ccl_vector_t>
TRACCC_HOST_DEVICE inline unsigned int find_root(const ccl_vector_t& L,
                                                 unsigned int o,
                                                 unsigned int N,
                                                 unsigned int e) {

    unsigned int r = e;
    assert(r < N);
    while (L[r+o] != r) {
        r = L[r+o];
        assert(r < N);
    }
    return r;
}

/// Create a union of two entries @param e1 and @param e2
///
/// @param L an equivalance table
///
/// @return the rleast common ancestor of the entries
template <typename ccl_vector_t>
TRACCC_HOST_DEVICE inline unsigned int make_union(ccl_vector_t& L,
                                                  unsigned int e1,
                                                  unsigned int e2) {

    int e;
    if (e1 < e2) {
        e = e1;
        assert(e2 < L.size());
        L[e2] = e;
    } else {
        e = e2;
        assert(e1 < L.size());
        L[e1] = e;
    }
    return e;
}

template <typename ccl_vector_t>
TRACCC_HOST_DEVICE inline unsigned int make_union(ccl_vector_t& L,
                                                  unsigned int o,
                                                  unsigned int N,
                                                  unsigned int e1,
                                                  unsigned int e2) {

    int e;
    if (e1 < e2) {
        e = e1;
        assert(e2 < N);
        L[e2+o] = e;
    } else {
        e = e2;
        assert(e1 < N);
        L[e1+o] = e;
    }
    return e;
}

/// Helper method to find adjacent cells
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolan to indicate 8-cell connectivity
TRACCC_HOST_DEVICE inline bool is_adjacent(traccc::cell a, traccc::cell b) {
    return (a.channel0 - b.channel0) * (a.channel0 - b.channel0) <= 1 and
           (a.channel1 - b.channel1) * (a.channel1 - b.channel1) <= 1;
}

TRACCC_HOST_DEVICE inline bool is_adjacent(unsigned int i0, unsigned int i1, 
                                           unsigned int j0, unsigned int j1) {
    return (i0 - j0) * (i0 - j0) <= 1 and
           (i1 - j1) * (i1 - j1) <= 1;
}

/// Helper method to find define distance,
/// does not need abs, as channels are sorted in
/// column major
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolan to indicate !8-cell connectivity
TRACCC_HOST_DEVICE inline bool is_far_enough(traccc::cell a, traccc::cell b) {
    return (a.channel1 - b.channel1) > 1;
}

TRACCC_HOST_DEVICE inline bool is_far_enough(unsigned int i1, unsigned int j1) {
    return (i1 - j1) > 1;
}

/// Sparce CCL algorithm
///
/// @param cells is the cell collection
/// @param L is the vector of the output indices (to which cluster a cell
/// belongs to)
/// @param labels is the number of clusters found
/// @return number of clusters
template <typename cell_container_t, typename VV>
TRACCC_HOST_DEVICE inline unsigned int sparse_ccl(const cell_container_t& cells,
                        std::size_t idx,
                        VV& channel0, VV& channel1,
                        VV& cumulsize, VV& moduleidx, VV& L) {

    unsigned int labels = 0;

    // The number of cells.
    unsigned int doffset = (idx == 0 ? 0 : cumulsize[idx -1]);
    const unsigned int n_cells = cumulsize[idx] - doffset;

    // first scan: pixel association
    unsigned int start_j = 0;
    for (unsigned int i = 0; i < n_cells; ++i) {
        L[i+doffset] = i;
        int ai = i;
        if (i > 0) {
            for (unsigned int j = start_j; j < i; ++j) {
                if (is_adjacent(channel0[i+doffset], channel1[i+doffset],
                                channel0[j+doffset], channel1[j+doffset])) {
                    ai = make_union(L, doffset, n_cells, ai, find_root(L, doffset, n_cells, j));
                } else if (is_far_enough(channel1[i+doffset], channel1[j+doffset])) {
                    ++start_j;
                }
            }
        }
    }

    // second scan: transitive closure
    for (unsigned int i = 0; i < n_cells; ++i) {
        unsigned int l = 0;
        if (L[i+doffset] == i) {
            ++labels;
            l = labels;
        } else {
            l = L[L[i+doffset]+doffset];
        }
        L[i+doffset] = l;
    }

    return labels;
}

template <typename cell_container_t, typename ccl_vector_t>
TRACCC_HOST_DEVICE inline unsigned int sparse_ccl(const cell_container_t cells,
                                                  ccl_vector_t& L) {

    unsigned int labels = 0;

    // The number of cells.
    const unsigned int n_cells = cells.size();

    // first scan: pixel association
    unsigned int start_j = 0;
    for (unsigned int i = 0; i < n_cells; ++i) {
        L[i] = i;
        int ai = i;
        if (i > 0) {
            for (unsigned int j = start_j; j < i; ++j) {
                if (is_adjacent(cells[i], cells[j])) {
                    ai = make_union(L, ai, find_root(L, j));
                } else if (is_far_enough(cells[i], cells[j])) {
                    ++start_j;
                }
            }
        }
    }

    // second scan: transitive closure
    for (unsigned int i = 0; i < n_cells; ++i) {
        unsigned int l = 0;
        if (L[i] == i) {
            ++labels;
            l = labels;
        } else {
            l = L[L[i]];
        }
        L[i] = l;
    }

    return labels;
}

}  // namespace detail

}  // namespace traccc