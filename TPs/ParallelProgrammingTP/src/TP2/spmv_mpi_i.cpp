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
    std::vector<int> nnz_counts(size, 0);
    int total_nnz, total_size;

    int data_count = 3 * size + 2; // 3 arrays with 'size' elements each
    std::vector<int> combined_buffer(data_count);
    if (rank == 0) {
      // Partition rows
      local_nrows = full_data.nrows / size;
      std::size_t remainder = full_data.nrows % size;

      std::fill(row_counts.begin(), row_counts.end(), local_nrows);
      for (int i = 0; i < remainder; ++i) row_counts[i]++;
      std::partial_sum(row_counts.begin(), row_counts.end(), row_displs.begin() + 1);
    
      for (int i = 0; i < size; ++i) {
        nnz_counts[i] = full_data.kcol[row_displs[i] + row_counts[i]] - full_data.kcol[row_displs[i]];
      }

      total_nnz = full_data.cols.size();
      total_size = full_data.kcol.size() + total_nnz * 2;

      combined_buffer[0] = total_nnz;
      combined_buffer[1] = total_size;

      std::copy(row_counts.begin(), row_counts.end(), combined_buffer.begin() + 2);
      std::copy(row_displs.begin(), row_displs.end(), combined_buffer.begin() + size + 2);
      std::copy(nnz_counts.begin(), nnz_counts.end(), combined_buffer.begin() + 2 * size + 2);

      
    }

    // Broadcast counts and displacements
    MPI_Bcast(combined_buffer.data(), data_count, MPI_INT, 0, comm);

    if (rank != 0) {

      total_nnz = combined_buffer[0];
      total_size = combined_buffer[1];
      row_counts.assign(combined_buffer.begin() + 2, combined_buffer.begin() + size + 2);
      row_displs.assign(combined_buffer.begin() + size + 2, combined_buffer.begin() + 2 * size + 2);
      nnz_counts.assign(combined_buffer.begin() + 2 * size + 2, combined_buffer.end());
    }

    std::vector<int> nnz_displs(size, 0);
    std::partial_sum(nnz_counts.begin(), nnz_counts.end() - 1, nnz_displs.begin() + 1);

    // Prepare local data
    local_data.nrows = row_counts[rank];
    local_data.kcol.resize(local_data.nrows + 1);
    local_data.cols.resize(nnz_counts[rank]);
    local_data.values.resize(nnz_counts[rank]);
    
    // Combined buffer for kcol, cols, values
    std::vector<char> total_combined_buffer(total_size * sizeof(int) + total_nnz * sizeof(double));

    if (rank == 0) {
        std::memcpy(total_combined_buffer.data(), full_data.kcol.data(), full_data.kcol.size() * sizeof(int));
        std::memcpy(total_combined_buffer.data() + full_data.kcol.size() * sizeof(int), full_data.cols.data(), full_data.cols.size() * sizeof(int));
        std::memcpy(total_combined_buffer.data() + (full_data.kcol.size() + full_data.cols.size()) * sizeof(int), full_data.values.data(), full_data.values.size() * sizeof(double));
    }

    std::vector<int> scatter_counts(size, 0);
    std::vector<int> scatter_displs(size, 0);

    if(rank == 0){
      // Compute scatter counts for each rank
      for (int i = 0; i < size; ++i) {
          scatter_counts[i] = (row_counts[i] + 1) * sizeof(int) // kcol
                            + nnz_counts[i] * sizeof(int)       // cols
                            + nnz_counts[i] * sizeof(double);   // values
      }

      // Compute displacements as prefix sum of scatter_counts
      for (int i = 1; i < size; ++i) {
          scatter_displs[i] = scatter_displs[i - 1] + scatter_counts[i - 1];
      }
    }
    
    // Scatter combined buffer
    std::vector<char> local_combined_buffer(row_counts[rank] * sizeof(int) + nnz_counts[rank] * (sizeof(int) + sizeof(double)));
    MPI_Scatterv(total_combined_buffer.data(), scatter_counts.data(), scatter_displs.data(), MPI_BYTE,
                 local_combined_buffer.data(), scatter_counts[rank], MPI_BYTE, 0, comm);

    // Unpack local data
    std::memcpy(local_data.kcol.data(), local_combined_buffer.data(), local_data.kcol.size() * sizeof(int));
    std::memcpy(local_data.cols.data(), local_combined_buffer.data() + local_data.kcol.size() * sizeof(int), local_data.cols.size() * sizeof(int));
    std::memcpy(local_data.values.data(), local_combined_buffer.data() + (local_data.kcol.size() + local_data.cols.size()) * sizeof(int), local_data.values.size() * sizeof(double));

    if (!local_data.kcol.empty()) {
        int initial = local_data.kcol[0];
        for (int& k : local_data.kcol) k -= initial;
        local_data.kcol[local_data.kcol.size() - 1] = nnz_counts[rank];
    }

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
