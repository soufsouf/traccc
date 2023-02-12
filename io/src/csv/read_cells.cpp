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

    // Construct the cell reader object.
    auto reader = make_cell_reader(filename);

    // Create cell counter vector.
    std::vector<cell_counter> cell_counts;
    cell_counts.reserve(5000);

    // Create a cell collection, which holds on to a flat list of all the cells.
    std::vector<csv::cell> allCells;
    allCells.reserve(50000);

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t10, t11;
    std::chrono::duration<double> t12;
    // Read all cells from input file.
    csv::cell iocell;
    t10 = std::chrono::high_resolution_clock::now();
    while (reader.read(iocell)) {
        t11 = std::chrono::high_resolution_clock::now();
        t12 += t11 - t10;

        // Hold on to this cell.
        allCells.push_back(iocell);

        // Increment the appropriate counter.
        auto rit = std::find_if(cell_counts.rbegin(), cell_counts.rend(),
                                [&iocell](const cell_counter& cc) {
                                    return cc.module == iocell.geometry_id;
                                });
        if (rit == cell_counts.rend()) {
            cell_counts.push_back({iocell.geometry_id, 1});
        } else {
            ++(rit->nCells);
        }
        t10 = std::chrono::high_resolution_clock::now();
    }
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // The number of modules that have cells in them.
    const std::size_t size = cell_counts.size();

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
            auto rit = std::find_if(
                result.get_headers().rbegin(), result.get_headers().rend(),
                [&iocell](const cell_module& module) {
                    return module.module == iocell.geometry_id;
                });
            assert(rit != result.get_headers().rend());
            last_module_index =
                std::distance(result.get_headers().begin(), rit.base()) - 1;
        }

        // Add the cell to the appropriate module.
        result.get_items()
            .at(last_module_index)
            .push_back({iocell.channel0, iocell.channel1, iocell.value,
                        iocell.timestamp});
    }

    // Do some post-processing on the cells.
    for (std::size_t i = 0; i < result.size(); ++i) {

        // Sort the cells of this module. (Not sure why this is needed. :-/)
        std::sort(result.get_items().at(i).begin(),
                  result.get_items().at(i).end(),
                  [](const traccc::cell& c1, const traccc::cell& c2) {
                      return c1.channel1 < c2.channel1;
                  });
    }

    std::cout <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t12).count() << " " <<
        std::endl;
    // Return the prepared object.
    return result;
}

cell_container_types::host
read_cells2(std::string_view filename,
                                       CellVec *cellsVec,
                                       ModuleVec *moduleVec,
                                      const geometry* geom,
                                      const digitization_config* dconfig,
                                      vecmem::memory_resource* mr) {
                                    
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    // Construct the cell reader object.
    auto reader = make_cell_reader(filename);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // Create cell counter vector.
    std::unordered_map<uint64_t, int> cellMap;
    std::vector<cell_counter> modules;
    modules.reserve(5000);

    // Create a cell collection, which holds on to a flat list of all the cells.
    std::vector<csv::cell> allCells;
    allCells.reserve(50000);

    std::chrono::high_resolution_clock::time_point t10, t11;
    std::chrono::duration<double> t12;

    // Read all cells from input file.
    csv::cell iocell;
    t10 = std::chrono::high_resolution_clock::now();
    while (reader.read(iocell)) {
        t11 = std::chrono::high_resolution_clock::now();
        t12 += t11 - t10;
        // Hold on to this cell.
        allCells.push_back(iocell);

        // Increment the appropriate counter.

        auto it = cellMap.find(iocell.geometry_id);
        if ( it == cellMap.end()) {
            cellMap.insert({iocell.geometry_id, modules.size()});
            modules.push_back({iocell.geometry_id, 1});
        } else {
            ++(modules[it->second].nCells);
        }
        t10 = std::chrono::high_resolution_clock::now();
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    // The number of modules that have cells in them.
    const std::size_t size = modules.size();
    const std::size_t allCellsCount = allCells.size();

    //cells = vecmem::data::vector_buffer<Cell>(1, m_mr.main);
    (*cellsVec).channel0 = int_vec(allCellsCount); //channel0
    (*cellsVec).channel1 = int_vec(allCellsCount); // channel1
    (*cellsVec).activation = scalar_vec(allCellsCount); // activation
    (*cellsVec).time = scalar_vec(allCellsCount); // time
    (*cellsVec).module_id = int_vec(allCellsCount); // module_id
    (*cellsVec).cluster_id = int_vec(allCellsCount); // cluster_id
    (*cellsVec).size = allCellsCount;

    (*moduleVec).cells_prefix_sum = int_vec(size);
    (*moduleVec).clusters_prefix_sum = int_vec(size);
    (*moduleVec).size = size;

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

    auto nCellsReader = [](cell_counter x) { return x.nCells; };
    auto sum = [](auto a, auto b) {return a + b;};
    std::transform_inclusive_scan(modules.begin(),
                        modules.end(),
                        (*moduleVec).cells_prefix_sum.begin(),
                        sum,
                        nCellsReader);

    for (int i = 0 ; i< 60 ; i++ )
                  printf("(*moduleVec).cells_prefix_sum: %u  \n", (*moduleVec).cells_prefix_sum[i] );
    
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();

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
        result.get_items().at(i).reserve(modules[i].nCells);

        // Construct the description of the detector module.
        cell_module& module = result.get_headers().at(i);
        module.module = modules[i].module;

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

    std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();

    // fill counter for cellsVec vectors
    //std::vector<unsigned int> module_fill_counter(size, 0);

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

       /* unsigned int midx = module_fill_counter[last_module_index];
        (*cellsVec).channel0[midx]   = iocell.channel0;
        (*cellsVec).channel1[midx]   = iocell.channel1;
        (*cellsVec).activation[midx] = iocell.value;
        (*cellsVec).time[midx]       = iocell.timestamp;
        (*cellsVec).module_id[midx] = last_module_index;*/

        // increment the fill counter for the current module
        //module_fill_counter[last_module_index]++;
    }

    std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();

    // Do some post-processing on the cells.
    unsigned int lb = 0;
    for (std::size_t i = 0; i < result.size(); ++i) {

        // Sort the cells of this module. (Not sure why this is needed. :-/)
        std::sort(result.get_items().at(i).begin(),
                  result.get_items().at(i).end(),
                  [](const traccc::cell& c1, const traccc::cell& c2) {
                      return c1.channel1 < c2.channel1;
                  });
        auto module_cells = result.at(i).items;
        lb = ( i == 0 ? 0 : (*moduleVec).cells_prefix_sum[i - 1]);
        unsigned int n_cells = (*moduleVec).cells_prefix_sum[i] - lb;
        for(std::size_t j = 0; j < n_cells; ++j) {
            (*cellsVec).channel0[lb+j]   = module_cells[j].channel0;
            (*cellsVec).channel1[lb+j]   = module_cells[j].channel1;
            (*cellsVec).activation[lb+j] = module_cells[j].activation;
            (*cellsVec).time[lb+j]       = module_cells[j].time;
            (*cellsVec).module_id[lb+j]  = i;
          // if(i ==0) std::cout<< "le channel 0 de module cell de j "<< j << " est : " <<  module_cells[j].channel0<< std::endl;

        }
    }

    std::chrono::high_resolution_clock::time_point t7 = std::chrono::high_resolution_clock::now();

    std::cout <<
        std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t12).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t5-t4).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count() << " " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t7-t6).count() << " " <<
        std::endl;

    // Return the prepared object.
    return result;
}

}  // namespace traccc::io::csv
