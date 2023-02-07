/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_cells.hpp"

#include "traccc/io/utils.hpp"

#include "make_cell_reader.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <algorithm>
#include <cassert>
#include <vector>
#include <numeric>

namespace {

/// Type used for counting the number of cells per detector module
struct cell_counter {
    uint64_t module = 0;
    std::size_t nCells = 0;
};

}  // namespace

namespace traccc::io::csv {

cell_container_types::host read_cells(std::string_view filename,
                                      const geometry* geom,
                                      const digitization_config* dconfig,
                                      vecmem::memory_resource* mr) {
                                    
    //std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    // Construct the cell reader object.
    auto reader = make_cell_reader(filename);

    // Create cell counter vector.
    std::unordered_map<uint64_t, int> cellMap;
    std::vector<cell_counter> cell_counts;
    cell_counts.reserve(5000);

    // Create a cell collection, which holds on to a flat list of all the cells.
    std::vector<csv::cell> allCells;
    allCells.reserve(50000);

    //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    // Read all cells from input file.
    csv::cell iocell;
    while (reader.read(iocell)) {

        // Hold on to this cell.
        allCells.push_back(iocell);

        // Increment the appropriate counter.

        auto it = cellMap.find(iocell.geometry_id);
        if ( it == cellMap.end()) {
            cellMap.insert({iocell.geometry_id, cell_counts.size()});
            cell_counts.push_back({iocell.geometry_id, 1});
        } else {
            ++(cell_counts[it->second].nCells);
        }
    }

    //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    // The number of modules that have cells in them.
    const std::size_t size = cell_counts.size();
    const std::size_t allCellsCount = allCells.size();

    // create Cell Vector
    using int_vec = vecmem::vector<unsigned int>;
    using scalar_vec = vecmem::vector<scalar>;

    //cells = vecmem::data::vector_buffer<Cell>(1, m_mr.main);
    CellVec cellsVec = {
        int_vec{allCellsCount, mr}, //channel0
        int_vec(allCellsCount, mr), // channel1
        scalar_vec(allCellsCount, mr), // activation
        scalar_vec(allCellsCount, mr), // time
        int_vec(allCellsCount, mr), // module_id
        int_vec(allCellsCount, mr) // cluster_id
    };

    //std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    // create prefix sum for modules size
    int_vec module_prefix_sum = int_vec{size, mr};

    auto nCellsReader = [](cell_counter x) { return x.nCells; };
    auto sum = [](auto a, auto b) {return a + b;};
    std::transform_inclusive_scan(cell_counts.begin(),
                        cell_counts.end(),
                        module_prefix_sum.begin(),
                        sum,
                        nCellsReader);

    //std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();

    // Construct the result container, and set up its headers.
    cell_container_types::host result;
    if (mr != nullptr) {
        result = cell_container_types::host{size, mr};
    } else {
        result = cell_container_types::host{
            cell_container_types::host::header_vector{size},
            cell_container_types::host::item_vector{size}};
    }
    for (std::size_t i = 0; i < size; ++i) {

        // Make sure that we would have just the right amount of space available
        // for the cells.
        result.get_items().at(i).reserve(cell_counts[i].nCells);

        // Construct the description of the detector module.
        cell_module& module = result.get_headers().at(i);
        module.module = cell_counts[i].module;

        // Find/set the 3D position of the detector module.
        if (geom != nullptr) {

            // Check if the module ID is known.
            if (!geom->contains(module.module)) {
                throw std::runtime_error(
                    "Could not find placement for geometry ID " +
                    std::to_string(module.module));
            }

            // Set the value on the module description.
            module.placement = (*geom)[module.module];
        }

        // Find/set the digitization configuration of the detector module.
        if (dconfig != nullptr) {

            // Check if the module ID is known.
            const digitization_config::Iterator geo_it =
                dconfig->find(module.module);
            if (geo_it == dconfig->end()) {
                throw std::runtime_error(
                    "Could not find digitization config for geometry ID " +
                    std::to_string(module.module));
            }

            // Set the value on the module description.
            const auto& binning_data = geo_it->segmentation.binningData();
            assert(binning_data.size() >= 2);
            module.pixel = {binning_data[0].min, binning_data[1].min,
                            binning_data[0].step, binning_data[1].step};
        }
    }

    //std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();

    // fill counter for cellsVec vectors
    std::vector<unsigned int> module_fill_counter(size, 0);

    // Now loop over all the cells, and put them into the appropriate modules.
    std::size_t last_module_index = 0;
    for (const csv::cell& iocell : allCells) {

        // Check if this cell belongs to the same module as the last cell did.
        if (iocell.geometry_id ==
            result.get_headers().at(last_module_index).module) {

            // If so, nothing needs to be done.
        }
        // If not, then it likely belongs to the next one.
        else if ((result.size() > (last_module_index + 1)) &&
                 (iocell.geometry_id ==
                  result.get_headers().at(last_module_index + 1).module)) {

            // If so, just increment the module index by one.
            ++last_module_index;
        }
        // If not that, then look for the appropriate module with a generic
        // search.
        else {
            auto it = cellMap.find(iocell.geometry_id);
            assert(it != cellMap.end());

            last_module_index = it->second;
        }

        // Add the cell to the appropriate module.
        result.get_items()
            .at(last_module_index)
            .push_back({iocell.channel0, iocell.channel1, iocell.value,
                        iocell.timestamp});

        unsigned int midx = module_fill_counter[last_module_index];
        cellsVec.channel0[midx]   = iocell.channel0;
        cellsVec.channel1[midx]   = iocell.channel1;
        cellsVec.activation[midx] = iocell.value;
        cellsVec.time[midx]       = iocell.timestamp;
        cellsVec.module_id[midx] = last_module_index;

        // increment the fill counter for the current module
        module_fill_counter[last_module_index]++;
    }

    //std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();

    // Do some post-processing on the cells.
    for (std::size_t i = 0; i < result.size(); ++i) {

        // Sort the cells of this module. (Not sure why this is needed. :-/)
        std::sort(result.get_items().at(i).begin(),
                  result.get_items().at(i).end(),
                  [](const traccc::cell& c1, const traccc::cell& c2) {
                      return c1.channel1 < c2.channel1;
                  });
    }

    /*std::chrono::high_resolution_clock::time_point t7 = std::chrono::high_resolution_clock::now();

    std::cout <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t5-t4).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t7-t6).count() << " " <<
        std::endl;*/

    // Return the prepared object.
    return result;
}

cell_container_types::host read_cells2(std::string_view filename,
                                      CellVec &cellsVec,
                                      const geometry* geom,
                                      const digitization_config* dconfig,
                                      vecmem::memory_resource* mr) {
                                    
    //std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    // Construct the cell reader object.
    auto reader = make_cell_reader(filename);
    // Create cell counter vector.
    std::unordered_map<uint64_t, int> cellMap;
    std::vector<cell_counter> cell_counts;
    cell_counts.reserve(5000);

    // Create a cell collection, which holds on to a flat list of all the cells.
    std::vector<csv::cell> allCells;
    allCells.reserve(50000);

    //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    // Read all cells from input file.
    csv::cell iocell;
    while (reader.read(iocell)) {

        // Hold on to this cell.
        allCells.push_back(iocell);

        // Increment the appropriate counter.

        auto it = cellMap.find(iocell.geometry_id);
        if ( it == cellMap.end()) {
            cellMap.insert({iocell.geometry_id, cell_counts.size()});
            cell_counts.push_back({iocell.geometry_id, 1});
        } else {
            ++(cell_counts[it->second].nCells);
        }
    }

    //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    // The number of modules that have cells in them.
    const std::size_t size = cell_counts.size();
    const std::size_t allCellsCount = allCells.size();

    // create Cell Vector
    using int_vec = vecmem::vector<unsigned int>;
    using scalar_vec = vecmem::vector<scalar>;

    //cells = vecmem::data::vector_buffer<Cell>(1, m_mr.main);
    cellsVec = {
        int_vec{allCellsCount, mr}, //channel0
        int_vec(allCellsCount, mr), // channel1
        scalar_vec(allCellsCount, mr), // activation
        scalar_vec(allCellsCount, mr), // time
        int_vec(allCellsCount, mr), // module_id
        int_vec(allCellsCount, mr) // cluster_id
    };

    //std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    // create prefix sum for modules size
    int_vec module_prefix_sum = int_vec{size, mr};

    auto nCellsReader = [](cell_counter x) { return x.nCells; };
    auto sum = [](auto a, auto b) {return a + b;};
    std::transform_inclusive_scan(cell_counts.begin(),
                        cell_counts.end(),
                        module_prefix_sum.begin(),
                        sum,
                        nCellsReader);

    //std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();

    // Construct the result container, and set up its headers.
    cell_container_types::host result;
    if (mr != nullptr) {
        result = cell_container_types::host{size, mr};
    } else {
        result = cell_container_types::host{
            cell_container_types::host::header_vector{size},
            cell_container_types::host::item_vector{size}};
    }
    for (std::size_t i = 0; i < size; ++i) {

        // Make sure that we would have just the right amount of space available
        // for the cells.
        result.get_items().at(i).reserve(cell_counts[i].nCells);

        // Construct the description of the detector module.
        cell_module& module = result.get_headers().at(i);
        module.module = cell_counts[i].module;

        // Find/set the 3D position of the detector module.
        if (geom != nullptr) {

            // Check if the module ID is known.
            if (!geom->contains(module.module)) {
                throw std::runtime_error(
                    "Could not find placement for geometry ID " +
                    std::to_string(module.module));
            }

            // Set the value on the module description.
            module.placement = (*geom)[module.module];
        }

        // Find/set the digitization configuration of the detector module.
        if (dconfig != nullptr) {

            // Check if the module ID is known.
            const digitization_config::Iterator geo_it =
                dconfig->find(module.module);
            if (geo_it == dconfig->end()) {
                throw std::runtime_error(
                    "Could not find digitization config for geometry ID " +
                    std::to_string(module.module));
            }

            // Set the value on the module description.
            const auto& binning_data = geo_it->segmentation.binningData();
            assert(binning_data.size() >= 2);
            module.pixel = {binning_data[0].min, binning_data[1].min,
                            binning_data[0].step, binning_data[1].step};
        }
    }

    //std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();

    // fill counter for cellsVec vectors
    std::vector<unsigned int> module_fill_counter(size, 0);

    // Now loop over all the cells, and put them into the appropriate modules.
    std::size_t last_module_index = 0;
    for (const csv::cell& iocell : allCells) {

        // Check if this cell belongs to the same module as the last cell did.
        if (iocell.geometry_id ==
            result.get_headers().at(last_module_index).module) {

            // If so, nothing needs to be done.
        }
        // If not, then it likely belongs to the next one.
        else if ((result.size() > (last_module_index + 1)) &&
                 (iocell.geometry_id ==
                  result.get_headers().at(last_module_index + 1).module)) {

            // If so, just increment the module index by one.
            ++last_module_index;
        }
        // If not that, then look for the appropriate module with a generic
        // search.
        else {
            auto it = cellMap.find(iocell.geometry_id);
            assert(it != cellMap.end());

            last_module_index = it->second;
        }

        // Add the cell to the appropriate module.
        result.get_items()
            .at(last_module_index)
            .push_back({iocell.channel0, iocell.channel1, iocell.value,
                        iocell.timestamp});

        unsigned int midx = module_fill_counter[last_module_index];
        cellsVec.channel0[midx]   = iocell.channel0;
        cellsVec.channel1[midx]   = iocell.channel1;
        cellsVec.activation[midx] = iocell.value;
        cellsVec.time[midx]       = iocell.timestamp;
        cellsVec.module_id[midx] = last_module_index;

        // increment the fill counter for the current module
        module_fill_counter[last_module_index]++;
    }

    //std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();

    // Do some post-processing on the cells.
    for (std::size_t i = 0; i < result.size(); ++i) {

        // Sort the cells of this module. (Not sure why this is needed. :-/)
        std::sort(result.get_items().at(i).begin(),
                  result.get_items().at(i).end(),
                  [](const traccc::cell& c1, const traccc::cell& c2) {
                      return c1.channel1 < c2.channel1;
                  });
    }

    /*std::chrono::high_resolution_clock::time_point t7 = std::chrono::high_resolution_clock::now();

    std::cout <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t5-t4).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t7-t6).count() << " " <<
        std::endl;*/

    // Return the prepared object.
    return result;
}

}  // namespace traccc::io::csv
