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

MPI_Datatype createCSRRangeType() {
    MPI_Datatype csr_type;
    int block_lengths[5] = {1, 1, 1, 1, 1}; // Each pointer is 1 unit
    MPI_Aint offsets[5];
    MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_UNSIGNED_LONG,  MPI_UNSIGNED_LONG};

    offsets[0] = offsetof(PPTP::CSRData, kcols);
    offsets[1] = offsetof(PPTP::CSRData, cols);
    offsets[2] = offsetof(PPTP::CSRData, values);
    offsets[3] = offsetof(PPTP::CSRData, nrows);
    offsets[4] = offsetof(PPTP::CSRData, nnz);

    MPI_Type_create_struct(5, block_lengths, offsets, types, &csr_type);
    MPI_Type_commit(&csr_type);

    return csr_type;
}

/*void recreateCSRMatrix(PPTP::CSRMatrix& matrix, const PPTP::CSRData& data) {
    // Resize vectors in the CSRMatrix
    matrix.m_kcol.resize(data.nrows + 1);
    matrix.m_cols.resize(data.nnz);
    matrix.m_values.resize(data.nnz);

    // Copy data from the received struct into the vectors
    std::copy(data.kcols, data.kcols + (data.nrows + 1), matrix.m_kcol.begin());
    std::copy(data.cols, data.cols + data.nnz, matrix.m_cols.begin());
    std::copy(data.values, data.values + data.nnz, matrix.m_values.begin());

    // Update matrix metadata
    matrix.nrows = data.nrows;
    matrix.ncols = *std::max_element(matrix.m_cols.begin(), matrix.m_cols.end()) + 1; // Max column index + 1
}*/

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
    std::cout << "Process " << world_rank + 1 << " in " << world_size <<std::endl;

    std::size_t global_nrows;
    std::vector<double> x;

    CSRMatrix local_matrix;
    std::size_t local_nrows;

    MPI_Datatype csr_type = createCSRRangeType();
    CSRData data;

    Timer::Sentry sentry(timer,"MPI_SpMV") ;
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
        int offset = 0;
        {
            std::size_t remainder = global_nrows % world_size;
            local_nrows = global_nrows / world_size + (world_rank < remainder ? 1:0);
            offset += local_nrows;

            for(std::size_t i = 1; i < world_size; ++i)
            {
                local_nrows = global_nrows / world_size + (i < remainder ? 1:0);

                // Sending local_nrows
                MPI_Send(&local_nrows, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
                
                data = matrix.extractSubmatrix(0, local_nrows).data();

                // Sending local matrix_data
                MPI_Send(&data, 1, csr_type, i, 1, MPI_COMM_WORLD);

                offset += local_nrows;
            }
        }
        // --------------------

        {
            std::vector<double> y(global_nrows);
            {
            Timer::Sentry sentry(timer,"SpMV") ;
            matrix.mult(x,y) ;
            }
            double normy = PPTP::norm2(y) ;
            std::cout<<"||y||="<<normy<<std::endl ;
        }
    } 

    // Step 4 : Zero Sending global matrix size and x and others Receiving 
    {
        // gloal_matrix_size
        MPI_Bcast(&global_nrows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        std::cout << "Global nrows " << global_nrows <<std::endl;

        // vector x
        x.resize(global_nrows);
        MPI_Bcast(x.data(), global_nrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::cout << "Vector of size " << x.size() << " last element " << x[-1] <<std::endl;
    }
    // --------------------

    // Step 6 : Receiving local matrix infos
    if(world_rank != 0)
    {
        MPI_Status status;

        // Receiving local_nrows
        MPI_Recv(&local_nrows, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &status);
        std::cout << "Received local nrows " << local_nrows <<std::endl;

        // Receiving local_matrix_data
        MPI_Recv(&data, 1, csr_type, 0, 1, MPI_COMM_WORLD, &status);
        std::cout << "Received local matrix data " << data.nrows <<std::endl;

    }
    // --------------------

    MPI_Type_free(&csr_type);

  }
  timer.printInfo();

  // step 2 : initialize (1) and finalize (2)
  MPI_Finalize(); // (2)
  // --------------------

  return 0 ;
}
