/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Katharina Kormann, Martin Kronbichler, Uppsala University,
 * 2009-2012, updated to MPI version with parallel vectors in 2016
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <iostream>
#include <fstream>
#include <sstream>


namespace Step37
{
  using namespace dealii;


  const unsigned int degree_finite_element = 2;
  const bool run_variable_sizes = true;


  template <int dim, int fe_degree, typename number>
  class LaplaceOperator : public MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >
  {
  public:
    typedef number value_type;

    LaplaceOperator ();

    void clear();

    virtual void compute_diagonal();

  private:
    virtual void apply_add(LinearAlgebra::distributed::Vector<number> &dst,
                           const LinearAlgebra::distributed::Vector<number> &src) const;

    void local_apply (const MatrixFree<dim,number>                     &data,
                      LinearAlgebra::distributed::Vector<number>       &dst,
                      const LinearAlgebra::distributed::Vector<number> &src,
                      const std::pair<unsigned int,unsigned int>       &cell_range) const;

    void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                                 LinearAlgebra::distributed::Vector<number>       &dst,
                                 const unsigned int                               &dummy,
                                 const std::pair<unsigned int,unsigned int>       &cell_range) const;
  };



  template <int dim, int fe_degree, typename number>
  LaplaceOperator<dim,fe_degree,number>::LaplaceOperator ()
    :
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number> >()
  {}



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::clear ()
  {
    MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >::clear();
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::local_apply (const MatrixFree<dim,number>                     &data,
                 LinearAlgebra::distributed::Vector<number>       &dst,
                 const LinearAlgebra::distributed::Vector<number> &src,
                 const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        phi.read_dof_values(src);
        phi.evaluate (false, true);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          phi.submit_gradient (phi.get_gradient(q), q);
        phi.integrate (false, true);
        phi.distribute_local_to_global (dst);
      }
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::apply_add (LinearAlgebra::distributed::Vector<number>       &dst,
               const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop (&LaplaceOperator::local_apply, this, dst, src);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::compute_diagonal ()
  {
    this->inverse_diagonal_entries.
    reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number> >());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    unsigned int dummy = 0;
    this->data->cell_loop (&LaplaceOperator::local_compute_diagonal, this,
                           inverse_diagonal, dummy);

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
      {
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) =
          1./inverse_diagonal.local_element(i);
      }
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::local_compute_diagonal (const MatrixFree<dim,number>               &data,
                            LinearAlgebra::distributed::Vector<number> &dst,
                            const unsigned int &,
                            const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);

    AlignedVector<VectorizedArray<number> > diagonal(phi.dofs_per_cell);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
              phi.submit_dof_value(VectorizedArray<number>(), j);
            phi.submit_dof_value(make_vectorized_array<number>(1.), i);

            phi.evaluate (false, true);
            for (unsigned int q=0; q<phi.n_q_points; ++q)
              phi.submit_gradient (phi.get_gradient(q), q);
            phi.integrate (false, true);
            diagonal[i] = phi.get_dof_value(i);
          }
        for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global (dst);
      }
  }




  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    void run (const bool do_adaptive);

  private:
    void setup_system ();
    void assemble_rhs ();
    void solve ();
    void output_results (const unsigned int cycle) const;
    void calculate_workload_imbalance ();

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim>  triangulation;
#else
    Triangulation<dim>                         triangulation;
#endif

    FE_Q<dim>                                  fe;
    DoFHandler<dim>                            dof_handler;

    ConstraintMatrix                           constraints;
    typedef LaplaceOperator<dim,degree_finite_element,double> SystemMatrixType;
    SystemMatrixType                           system_matrix;

    MGConstrainedDoFs                          mg_constrained_dofs;
    typedef LaplaceOperator<dim,degree_finite_element,float>  LevelMatrixType;
    MGLevelObject<LevelMatrixType>             mg_matrices;

    LinearAlgebra::distributed::Vector<double> solution;
    LinearAlgebra::distributed::Vector<double> system_rhs;

    ConditionalOStream                         pcout;
    ConditionalOStream                         time_details;
  };



  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem ()
    :
#ifdef DEAL_II_WITH_P4EST
    triangulation(MPI_COMM_WORLD,
                  Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
#else
    triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
#endif
    fe (degree_finite_element),
    dof_handler (triangulation),
    pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_details (std::cout, true &&
                  Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}



  template <typename StreamType>
  void print_averages(const std::string &name,
		      Timer      &timer,
		      StreamType &stream)
  {
    Utilities::MPI::MinMaxAvg timings = Utilities::MPI::min_max_avg(timer.wall_time(), MPI_COMM_WORLD);
    stream << std::setw(30) << name
	   << std::setw(12) << std::left << timings.min
	   << "(p" << std::setw(5) << timings.min_index << ")  "
	   << std::setw(12) << std::left << timings.avg
	   << std::setw(12) << std::left << timings.max
	   << "(p" << std::setw(5) << timings.max_index << ")"
	   << std::endl;

    timer.restart();
  }


  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {
    Timer time;
    time.start ();

    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs (fe);

    print_averages("Distribute DoFs", time, time_details);

    dof_handler.distribute_mg_dofs (fe);

    print_averages("Distribute level DoFs", time, time_details);

    pcout << "Number of active mesh cells:  "
          << triangulation.n_global_active_cells()
          << std::endl;
    pcout << "Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << std::endl;

    time.restart();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(),
                                              constraints);
    constraints.close();
    print_averages("Build constraints", time, time_details);

    {
      typename MatrixFree<dim,double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim,double>::AdditionalData::none;
      additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                              update_quadrature_points);
      std::shared_ptr<MatrixFree<dim,double> >
      system_mf_storage(new MatrixFree<dim,double>());
      system_mf_storage->reinit (dof_handler, constraints, QGauss<1>(fe.degree+1),
                                 additional_data);
      system_matrix.initialize (system_mf_storage);
    }

    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(system_rhs);

    print_averages("Initialize system mf", time, time_details);

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels-1);

    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

    for (unsigned int level=0; level<nlevels; ++level)
      {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level,
                                                      relevant_dofs);
        ConstraintMatrix level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();

        typename MatrixFree<dim,float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim,float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                                update_quadrature_points);
        additional_data.level_mg_handler = level;
        std::shared_ptr<MatrixFree<dim,float> >
        mg_mf_storage_level(new MatrixFree<dim,float>());
        mg_mf_storage_level->reinit(dof_handler, level_constraints,
                                    QGauss<1>(fe.degree+1), additional_data);

        mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs,
                                      level);
      }
    print_averages("Initialize level mf", time, time_details);
  }




  template <int dim>
  void LaplaceProblem<dim>::assemble_rhs ()
  {
    Timer time;

    system_rhs = 0;
    FEEvaluation<dim,degree_finite_element> phi(*system_matrix.get_matrix_free());
    for (unsigned int cell=0; cell<system_matrix.get_matrix_free()->n_macro_cells(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          phi.submit_value(make_vectorized_array<double>(1.0), q);
        phi.integrate(true, false);
        phi.distribute_local_to_global(system_rhs);
      }
    system_rhs.compress(VectorOperation::add);

    print_averages("Assemble RHS vector", time, time_details);
  }




  template <int dim>
  void LaplaceProblem<dim>::solve ()
  {
    Timer time;
    MGTransferMatrixFree<dim,float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    print_averages("Initialize MG transfer", time, time_details);

    typedef PreconditionChebyshev<LevelMatrixType,LinearAlgebra::distributed::Vector<float> > SmootherType;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float> >
    mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level = 0; level<triangulation.n_global_levels(); ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range = 15.;
            smoother_data[level].degree = 4;
            smoother_data[level].eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
      }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float> > mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<float> > mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType> > mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<float> > mg_interface(mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float> > mg(mg_matrix,
                                                             mg_coarse,
                                                             mg_transfer,
                                                             mg_smoother,
                                                             mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>,
                   MGTransferMatrixFree<dim,float> >
                   preconditioner(dof_handler, mg, mg_transfer);


    SolverControl solver_control (100, 1e-12*system_rhs.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double> > cg (solver_control);
    print_averages("Initialize MG smoothers", time, time_details);

    for (unsigned int i=0; i<5; ++i)
      {
        time.restart();
        preconditioner.vmult(solution, system_rhs);
	print_averages("Time V-cycle precondition", time, pcout);
      }

    for (unsigned int i=0; i<5; ++i)
      {
        time.restart();
        system_matrix.vmult(solution, system_rhs);
	print_averages("Time matrix * vector", time, pcout);
      }


    for (unsigned int i=0; i<3; ++i)
      {
        solution  = 0;
        cg.solve (system_matrix, solution, system_rhs,
                  preconditioner);

	print_averages("Time solve (" + std::to_string(solver_control.last_step()) + " iterations)", time, pcout);
      }

    constraints.distribute(solution);
  }




  template <int dim>
  void LaplaceProblem<dim>::output_results (const unsigned int cycle) const
  {
    if (triangulation.n_global_active_cells() > 1000000)
      return;

    DataOut<dim> data_out;

    solution.update_ghost_values();
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();

    std::ostringstream filename;
    filename << "solution-"
             << cycle
             << "." << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
             << ".vtu";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          {
            std::ostringstream filename;
            filename << "solution-"
                     << cycle
                     << "."
                     << i
                     << ".vtu";

            filenames.push_back(filename.str().c_str());
          }
        std::string master_name = "solution-" + Utilities::to_string(cycle) + ".pvtu";
        std::ofstream master_output (master_name.c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }
  }

  template <int dim>
  void LaplaceProblem<dim>::calculate_workload_imbalance ()
  {
      unsigned int n_proc = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      unsigned int n_global_levels = triangulation.n_global_levels();

      unsigned long long int work_estimate = 0;
      unsigned long long int total_cells_in_hierarchy = 0;

      for (unsigned int lvl=0; lvl<n_global_levels; ++lvl)
      {
          unsigned long long int max_num_owned_on_lvl;
          unsigned long long int total_cells_on_lvl;
          unsigned long long int n_owned_cells_on_lvl = 0;

          typename Triangulation<dim>::cell_iterator
                  cell = triangulation.begin(lvl),
                  endc = triangulation.end(lvl);
          for (; cell!=endc; ++cell)
          {
              if (cell->is_locally_owned_on_level())
                  n_owned_cells_on_lvl += 1;
          }

          MPI_Reduce(&n_owned_cells_on_lvl, &max_num_owned_on_lvl, 1,
                     MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
                     MPI_COMM_WORLD);
          //Work estimated by summing up max number of cells on each level
          work_estimate += max_num_owned_on_lvl;

          MPI_Reduce(&n_owned_cells_on_lvl, &total_cells_on_lvl, 1,
                     MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
                     MPI_COMM_WORLD);
          total_cells_in_hierarchy += total_cells_on_lvl;
      }
      double ideal_work = total_cells_in_hierarchy / (double)n_proc;
      double workload_imbalance_ratio = work_estimate / ideal_work;

      pcout << "Workload Imbalance Ratio: " << workload_imbalance_ratio << std::endl;
  }

  template <int dim>
  void LaplaceProblem<dim>::run (const bool do_adaptive)
  {
    pcout << "Number of MPI processes: " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;
    pcout << "Testing " << fe.get_name() << std::endl << std::endl;
    unsigned int n_cycles = 5;
    unsigned int sizes [] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128};
    if (run_variable_sizes)
      n_cycles = sizeof(sizes)/sizeof(unsigned int);
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << std::endl;
        if (run_variable_sizes)
          {
            Timer time;
            triangulation.clear();
            unsigned int n_refinements = 0;
            unsigned int n_subdiv = sizes[cycle];
            if (n_subdiv > 1)
              while (n_subdiv%2 == 0)
                {
                  n_refinements += 1;
                  n_subdiv /= 2;
                }
            if (dim == 2)
              n_refinements += 3;
            unsigned int njobs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
            while (njobs > 0)
              {
                njobs >>= dim;
                n_refinements++;
              }
            GridGenerator::subdivided_hyper_cube (triangulation, n_subdiv, -1, 1);
            triangulation.refine_global(n_refinements);
            pcout << "Time grid generation uniform:  " << time.wall_time() << "s" << std::endl;
            if (do_adaptive)
              {
                time.restart();
                for (typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
                  if (cell->is_locally_owned() &&
                      cell->center().norm() < 0.55)
                    cell->set_refine_flag();
                triangulation.execute_coarsening_and_refinement();
                for (typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
                  if (cell->is_locally_owned() &&
                      cell->center().norm() > 0.3 && cell->center().norm() < 0.42)
                    cell->set_refine_flag();
                triangulation.execute_coarsening_and_refinement();
                for (typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
                  if (cell->is_locally_owned() &&
                      cell->center().norm() > 0.335 && cell->center().norm() < 0.39)
                    cell->set_refine_flag();
                triangulation.execute_coarsening_and_refinement();
                pcout << "Time grid generation adaptive: " << time.wall_time() << "s" << std::endl;
              }

          }
        else
          {
            if (cycle == 0)
              {
                GridGenerator::hyper_cube (triangulation, -1, 1.);
                //triangulation.refine_global (3-dim);
              }
            triangulation.refine_global (1);
            //triangulation.begin_active()->set_refine_flag();
            //triangulation.execute_coarsening_and_refinement();
          }

        setup_system ();
        assemble_rhs ();
        solve ();
        //output_results (cycle);
        calculate_workload_imbalance ();
        pcout << std::endl;
        const types::global_dof_index thres = do_adaptive ? 2000000 : 2700000;
        if (dof_handler.n_dofs() > thres*Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
          break;
      };
  }
}




int main (int argc, char *argv[])
{
  try
    {
      using namespace Step37;

      Timer time;
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
      print_averages("MPI_Init", time, pcout);

      {
	pcout << std::endl
	      << "------------------------------------------------------------------"
	      << std::endl << std::endl;
	pcout << "Running uniform refinement in 3D" << std::endl;
	LaplaceProblem<3> laplace_problem;
	laplace_problem.run (false);

	pcout << std::endl
	      << "------------------------------------------------------------------"
	      << std::endl << std::endl;
	pcout << "Running adaptive refinement in 3D" << std::endl;
	laplace_problem.run (true);
      }
      {
	pcout << std::endl
	      << "------------------------------------------------------------------"
	      << std::endl << std::endl;
	pcout << "Running uniform refinement in 2D" << std::endl;
	LaplaceProblem<2> laplace_problem;
	laplace_problem.run (false);

	pcout << std::endl
	      << "------------------------------------------------------------------"
	      << std::endl << std::endl;
	pcout << "Running adaptive refinement in 2D" << std::endl;
	laplace_problem.run (true);
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
