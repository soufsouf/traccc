/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/alt_cell.hpp"

// System include(s).
#include <cstddef>
#include <cuda_runtime_api.h>
namespace traccc::device {

/// Function for looking for adjacent cells. The cell ids will range from 0 to
/// max_cells_per_partition and the number of blocks will equal the number of
/// partitions, hence checking all cells.
///
/// @param[in] cells    Collection of cells
/// @param[in] cid      Current cell id
/// @param[in] start    Current partition start point
/// @param[in] end      Current partition end point
/// @param[out] ajc     Number of adjacent cells
/// @param[out] ajv     Indices of adjacent cells
///
TRACCC_DEVICE


inline void reduce_problem_cell2(
    const traccc::alt_cell* cells_device,
    const unsigned short cid, const unsigned int start, const unsigned int end,
    unsigned char& adjc, unsigned short adjv[8],unsigned short* id_fathers);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/reduce_problem_cell.ipp"