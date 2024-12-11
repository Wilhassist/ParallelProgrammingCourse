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
    int rank, int size, 
    MPI_Comm comm
) {

    std::vector<int> row_counts, row_displs;
    std::size_t local_nrows;
    // Partition rows among processes
    if (rank == 0){
      local_nrows = full_data.nrows / size;
      std::size_t remainder = full_data.nrows % size;

      row_counts.resize(size, local_nrows);
      for (int i = 0; i < remainder; ++i) row_counts[i]++;  // Handle extra rows

      row_displs.resize(size, 0);
      std::partial_sum(row_counts.begin(), row_counts.end() - 1, row_displs.begin() + 1);
    }
    
    row_counts.resize(size, 0);
    row_displs.resize(size, 0);
    
    // Broadcast row_counts and row_displs to all ranks
    MPI_Bcast(row_counts.data(), size, MPI_INT, 0, comm);
    MPI_Bcast(row_displs.data(), size, MPI_INT, 0, comm);

    // Prepare local row pointers
    local_data.nrows = row_counts[rank];
    local_data.kcol.resize(local_data.nrows + 1);

    /*if (rank == 0) {
      // Root process sends data to each process
      for (int i = 0; i < size; ++i) {
          int send_count = row_counts[i] + 1; // Number of elements to send

          std::cout << "iteration " << i << " + " << send_count << std::endl;
          MPI_Send(full_data.kcol.data() + row_displs[i], send_count, MPI_INT, i, 0, comm);
      }
    } else {
      // Other processes receive their data
      int recv_count = row_counts[rank] + 1; // Number of elements to receive
      std::cout << "rank " << rank << " + " << recv_count << std::endl;
      
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

    // Adjust local row pointers
    /*int row_offset = full_data.kcol[row_displs[rank]];
    for (int& k : local_data.kcol) {
        k -= row_offset;
    }*/

    // Calculate non-zero elements for each process
    /*std::vector<int> nnz_counts(size, 0);
    for (int i = 0; i < size; ++i) {
        nnz_counts[i] = full_data.kcol[row_displs[i] + row_counts[i]] - full_data.kcol[row_displs[i]];
    }

    for (int i = 0; i < size; ++i) {
    std::cout << "Process " << i << ": "
              << "kcol count = " << row_counts[i] + 1 << ", "
              << "cols count = " << nnz_counts[i] << ", " 
              << "values count = " << nnz_counts[i] 
              << std::endl;
    }

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

    // Update local nnz count
    local_data.nnz = nnz_counts[rank];*/
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

    CSRData full_data, local_data;
    std::size_t global_nrows;
    std::vector<double> x, local_y;

    if(world_rank == 0)
    {
      CSRMatrix matrix ;
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
      //MPI_Bcast(&global_nrows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

      // vector x
      x.resize(global_nrows);
      //MPI_Bcast(x.data(), global_nrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    scatterCSRMatrix(full_data, local_data, world_rank, world_size, MPI_COMM_WORLD);
    

    // Verify results
    std::cout << "Rank " << world_rank << " local_kcol: ";
    for (const auto& k : local_data.kcol) {
        std::cout << k << " ";
    }
    std::cout << std::endl;
    //std::cout << "\n";

    std::cout << "Rank " << world_rank << " local_cols: ";
    std::cout << local_data.cols.size() << " ";
    std::cout << "\n";

    std::cout << "Rank " << world_rank << " local_values: ";
    std::cout << local_data.values.size() << " ";
    std::cout << "\n";

  }


  if(world_rank == 0)
    timer.printInfo();

  MPI_Finalize();
  return 0 ;
}
