/*
 * helloworld.cpp
 *
 *  Created on: Aug 16, 2018
 *      Author: gratienj
 */

#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

// step 1 : include mpi
#include <mpi.h>
// --------------------

#include <string>
#include <vector>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/LU>

#include "MatrixVector/CSRMatrix.h"
#include "MatrixVector/LinearAlgebra.h"
#include "MatrixVector/MatrixGenerator.h"

#include "Utils/Timer.h"

int main(int argc, char** argv)
{
  using namespace boost::program_options ;
  options_description desc;
  desc.add_options()
      ("help", "produce help")
      ("nb-threads",value<int>()->default_value(0), "nb threads")
      ("nrows",value<int>()->default_value(0), "matrix size")
      ("nx",value<int>()->default_value(0), "nx grid size")
      ("file",value<std::string>(), "file input")
      ("eigen",value<int>()->default_value(0), "use eigen package") ;
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  if (vm.count("help"))
  {
      std::cout << desc << "\n";
      return 1;
  }

  // step 2 : initialize (1) and finalize (2)
  MPI_Init(&argc, &argv); // (1)
  // --------------------

  // step 3 : Initialize Variables
  int world_size;
  int world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  // --------------------

  using namespace PPTP ;

  Timer timer ;
  MatrixGenerator generator ;

  std::size_t global_nrows;
  std::vector<double> x, y;

  CSRMatrix local_matrix;

  // Step 3: Define sendcounts and displacements
  std::vector<int> sendcounts(world_size, 0);
  std::vector<int> displacements(world_size, 0);

  if(world_rank == 0)
  {

    CSRMatrix matrix;

    if(vm.count("file"))
    {
      std::string file = vm["file"].as<std::string>() ;
      generator.readFromFile(file,matrix) ;
    }
    else
    {
      int nx = vm["nx"].as<int>() ;
      generator.genLaplacian(nx,matrix) ;
    }
        
    global_nrows = matrix.nrows();
    x.resize(global_nrows);

    for(std::size_t i=0;i<global_nrows;++i)
      x[i] = i+1 ;

    // Step 5 : Zero Sending local matrix info
    Timer::Sentry sentry(timer,"MPISpMV") ;
    // Calculate sendcounts and displacements
    std::size_t remainder = global_nrows % world_size;
    for (int i = 0; i < world_size; ++i)
    {
      sendcounts[i] = global_nrows / world_size + (i < remainder ? 1 : 0);
      if (i > 0)
        displacements[i] = displacements[i - 1] + sendcounts[i - 1];
    }

    // Flatten the matrix into a 1D buffer for scattering
    std::vector<double> flattened_matrix = matrix.flattenForScatter();
    MPI_Scatterv(flattened_matrix.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE,
          nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  // --------------------
     
  std::cout << "Process " << world_rank + 1 << " in " << world_size <<std::endl;

  // Step 4: Broadcast global_nrows and vector x
  MPI_Bcast(&global_nrows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  x.resize(global_nrows);
  MPI_Bcast(x.data(), global_nrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // --------------------

  // Step 5: Receive local matrix using MPI_Scatterv
  int local_nrows = sendcounts[world_rank];
  std::vector<double> local_data(local_nrows);
  MPI_Scatterv(nullptr, sendcounts.data(), displacements.data(), MPI_DOUBLE,
            local_data.data(), local_nrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  local_matrix.copyCSRMatrixFromData(local_data);

  // Step 6: Compute local multiplication
  std::vector<double> local_y(local_matrix.nrows());
  local_matrix.mult(x, local_y);

  // Step 7: Gather results back to process 0
  y.resize(global_nrows);
  MPI_Gatherv(local_y.data(), local_y.size(), MPI_DOUBLE,
            y.data(), sendcounts.data(), displacements.data(),
            MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank == 0)
  {
    double normy2 = PPTP::norm2(y);
    std::cout << "||MPI - y||=" << normy2 << std::endl;
  }

  timer.printInfo();

  // step 2 : initialize (1) and finalize (2)
  MPI_Finalize(); // (2)
  // --------------------
  return 0 ;
}
