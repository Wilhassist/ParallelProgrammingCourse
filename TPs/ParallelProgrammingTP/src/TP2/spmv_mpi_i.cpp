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
#include <numeric>


#include "MatrixVector/CSRMatrix.h"
#include "MatrixVector/LinearAlgebra.h"
#include "MatrixVector/MatrixGenerator.h"

#include "Utils/Timer.h"

void scatterCSRMatrix(
    const PPTP::CSRData& full_data, 
    PPTP::CSRData& local_data, 
    std::vector<int>& row_counts, std::vector<int>& row_displs,
    int rank, int size, 
    MPI_Comm comm
) {
    std::size_t local_nrows;
    

    if (rank == 0) {
        // Partition rows
        local_nrows = full_data.nrows / size;
        std::size_t remainder = full_data.nrows % size;

        std::fill(row_counts.begin(), row_counts.end(), local_nrows);
        for (int i = 0; i < remainder; ++i) row_counts[i]++;
        std::partial_sum(row_counts.begin(), row_counts.end() - 1, row_displs.begin() + 1);
    }

    // Broadcast counts and displacements
    MPI_Bcast(row_counts.data(), size, MPI_INT, 0, comm);
    MPI_Bcast(row_displs.data(), size, MPI_INT, 0, comm);

    // Prepare local data
    local_data.nrows = row_counts[rank];
    local_data.kcol.resize(local_data.nrows + 1);

    /*if (rank == 0) {
        // Root process sends data to each process
        for (int i = 0; i < size; ++i) {
            int send_count = row_counts[i] + 1; // Number of elements to send
            std::cout << "Process " << i << " send count: " << send_count << std::endl;
            MPI_Send(full_data.kcol.data() + row_displs[i], send_count, MPI_INT, i, 0, comm);
        }
    } else {
        // Other processes receive their data
        int recv_count = row_counts[rank] + 1; // Number of elements to receive
        std::cout << "Process " << rank << " recv count: " << recv_count << std::endl;

        local_data.kcol.resize(recv_count); // Proper resizing to avoid truncation
        MPI_Recv(local_data.kcol.data(), recv_count, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
    }*/

    
    MPI_Scatterv(
        full_data.kcol.data(), 
        row_counts.data(), 
        row_displs.data(), 
        MPI_INT, 
        local_data.kcol.data(), 
        row_counts[rank] + 1, 
        MPI_INT, 
        0, comm
    );

    if (rank == 0) {
      std::cout << "local kcol: ";
      for (const auto& val : local_data.kcol) {
          std::cout << val << " ";
      }
      std::cout << std::endl;
    }

    // Adjust row pointers
    std::vector<int> row_offsets(size, 0);

    // Compute offsets on rank 0
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            if (row_displs[i] < full_data.kcol.size()) {
                row_offsets[i] = full_data.kcol[row_displs[i]];
            } else {
                row_offsets[i] = 0; // Fallback in case of invalid displacement
            }
        }
    }

    int local_row_offset = 0;
    MPI_Scatter(row_offsets.data(), 1, MPI_INT, &local_row_offset, 1, MPI_INT, 0, comm);

    if (!local_data.kcol.empty()) {
        for (int& k : local_data.kcol) k -= local_row_offset;
    }

    // Calculate nnz counts
    std::vector<int> nnz_counts(size, 0);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            nnz_counts[i] = full_data.kcol[row_displs[i] + row_counts[i]] - full_data.kcol[row_displs[i]];
        }
    }
    MPI_Bcast(nnz_counts.data(), size, MPI_INT, 0, comm);

    /*if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            std::cout << "Process " << i << ": "
                      << "kcol count = " << row_counts[i] + 1 << ", "
                      << "cols count = " << nnz_counts[i] << ", "
                      << "values count = " << nnz_counts[i]
                      << std::endl;
        }
    }*/

    std::vector<int> nnz_displs(size, 0);
    std::partial_sum(nnz_counts.begin(), nnz_counts.end() - 1, nnz_displs.begin() + 1);

    // Scatter columns and values
    local_data.cols.resize(nnz_counts[rank]);
    local_data.values.resize(nnz_counts[rank]);

    MPI_Scatterv(
        full_data.cols.data(), 
        nnz_counts.data(), 
        nnz_displs.data(), 
        MPI_INT, 
        local_data.cols.data(), 
        nnz_counts[rank], 
        MPI_INT, 
        0, comm
    );

    MPI_Scatterv(
        full_data.values.data(), 
        nnz_counts.data(), 
        nnz_displs.data(), 
        MPI_DOUBLE, 
        local_data.values.data(), 
        nnz_counts[rank], 
        MPI_DOUBLE, 
        0, comm
    );

    // Update nnz count
    local_data.nnz = nnz_counts[rank];
}


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

  MPI_Init(&argc, &argv); 

  int world_size;
  int world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  using namespace PPTP ;

  Timer timer ;
  MatrixGenerator generator ;
  if(vm["eigen"].as<int>()==1)
  {
    typedef Eigen::SparseMatrix<double> MatrixType ;
    typedef Eigen::VectorXd             VectorType ;
    MatrixType matrix ;
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


    std::size_t nrows = matrix.rows();
    VectorType x(nrows);

    for(std::size_t i=0;i<nrows;++i)
      x(i) = i+1 ;

    VectorType y ;
    {
      Timer::Sentry sentry(timer,"EigenSpMV") ;
       y = matrix*x ;
    }

    double normy = PPTP::norm2(y) ;
    std::cout<<"||y||="<<normy<<std::endl ;
  }
  else
  {

    CSRMatrix local_matrix;
    CSRData full_data, local_data;
    std::size_t global_nrows;
    std::vector<double> x;

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
      std::vector<double> y;
      x.resize(global_nrows);
      y.resize(global_nrows);

      for(std::size_t i=0;i<global_nrows;++i)
        x[i] = i+1 ;

      {
        Timer::Sentry sentry(timer,"SpMV") ;
        matrix.mult(x,y) ;
      }
      double normy = PPTP::norm2(y) ;
      std::cout<<"||y||="<<normy<<std::endl ;

      Timer::Sentry sentry(timer,"MPI_SpMV") ;
      
      full_data = matrix.data();

    }

    
    {
      // gloal_matrix_size
      MPI_Bcast(&global_nrows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

      // vector x
      x.resize(global_nrows);
      MPI_Bcast(x.data(), global_nrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    std::vector<int> row_counts(world_size, 0), row_displs(world_size, 0);

    scatterCSRMatrix(full_data, local_data,row_counts,
                   row_displs, world_rank, world_size, MPI_COMM_WORLD);

    local_matrix.copyCSRMatrixFromCSRData(local_data);
    
    // Step 7 : Computing the local multiplication
    std::vector<double> local_y(local_matrix.nrows());
    {
      local_matrix.mult(x,local_y);
    }

    for (int p = 0; p < world_size; ++p) {
        if (world_rank == p) {
            std::cout << "Rank " << world_rank << " local_y: ";
            for (const auto& val : local_y) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD); // Synchronize output
    }

    

    // Gather the results back to process 0
    std::vector<double> y;
    if (world_rank == 0) {
        y.resize(full_data.nrows);  // Resize on rank 0 to hold the entire result
    }

    MPI_Gatherv(
        local_y.data(),             // Local buffer
        local_y.size(),             // Number of elements to send
        MPI_DOUBLE,                 // Data type
        y.data(),                   // Global buffer (on rank 0)
        row_counts.data(),          // Counts of rows per process
        row_displs.data(),          // Displacements
        MPI_DOUBLE,                 // Data type
        0,                          // Root process
        MPI_COMM_WORLD                       // Communicator
    );

    if (world_rank == 0) {
      std::cout << "Final gathered y: ";
      for (const auto& val : y) {
          std::cout << val << " ";
      }
      std::cout << std::endl;
    }

    if (world_rank == 0)
    {
      double normy2 = PPTP::norm2(y);
      std::cout<<"||MPI - y||="<<normy2<<std::endl;
    }
  }


  if(world_rank == 0)
    timer.printInfo();

  MPI_Finalize();
  return 0 ;
}
