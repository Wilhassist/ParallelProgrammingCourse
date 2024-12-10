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

    std::size_t global_nrows;
    std::vector<double> x;

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
      std::vector<double> y, local_y ;
      x.resize(global_nrows) ;
      y.resize(global_nrows) ;

      for(std::size_t i=0;i<global_nrows;++i)
        x[i] = i+1 ;
      {
        {
          Timer::Sentry sentry(timer,"SpMV") ;
          matrix.mult(x,y) ;
        }
        double normy = PPTP::norm2(y) ;
        std::cout<<"||y||="<<normy<<std::endl ;
      }

      Timer::Sentry sentry(timer,"MPI_SpMV") ;
      {

      }
    }


    {
      // gloal_matrix_size
      MPI_Bcast(&global_nrows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      std::cout << "Global nrows " << global_nrows <<std::endl;

      // vector x
      x.resize(global_nrows);
      MPI_Bcast(x.data(), global_nrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      std::cout << "Vector of size " << x.size() <<std::endl;
    }

    std::cout << "Process " << world_rank + 1 << " in " << world_size <<std::endl;
  }


  if(world_rank == 0)
    timer.printInfo();

  MPI_Finalize();
  return 0 ;
}
