// ---------------------------------------------------------------------
// $Id: step-39.cc 32156 2014-01-03 13:54:05Z heister $
//
// Copyright (C) 2010 - 2013 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

/*
 * Author: Guido Kanschat, Timo Heister
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/vector_slice.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/grad_div.h>

#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/generic_linear_algebra.h>

#ifdef DEAL_II_WITH_TRILINOS
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <ml_epetra_utils.h>
#include <ml_include.h>
#include <ml_MultiLevelPreconditioner.h>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS
#endif

#ifdef DEAL_II_WITH_PETSC
#include <petscsys.h>
#endif

namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

const double penalty_pre_factor = 2.0;

#include <iostream>
#include <fstream>

namespace Step39
{
  using namespace dealii;


  struct ProblemType
  {
    enum Kind
    {
      hyper_cube, hyper_L, cylinder, cylinder_shell, three_cylinders
    };
    static Kind parse(const std::string &name)
    {
      if (name == "hyper_cube")
        return hyper_cube;
      else if (name == "hyper_L")
        return hyper_L;
      else if (name == "cylinder")
        return cylinder;
      else if (name == "cylinder_shell")
        return cylinder_shell;
      else if (name == "three_cylinders")
        return three_cylinders;
      else
        AssertThrow(false, ExcNotImplemented());
      return hyper_cube;
    }
    static std::string get_options()
    {
      return "hyper_cube|hyper_L|cylinder|cylinder_shell|three_cylinders";
    }
  };


  template <int dim>
  class EllipseManifold : public Manifold<dim,dim>
  {
  public:
    EllipseManifold(const Point<dim> &axis,
                    const double main_radius,
                    const double cut_radius)
      :
      direction(axis),
      main_radius(main_radius),
      cut_radius(cut_radius)
    {
      // currently we assume to be in the plane x=0.

    }

    std::unique_ptr<Manifold<dim, dim>> clone() const
    {
      return std::make_unique<EllipseManifold<dim>>(direction, main_radius, cut_radius);
    }

    virtual Point<dim> get_intermediate_point(const Point<dim> &p1,
                                              const Point<dim> &p2,
                                              const double w) const
    {
//      if (std::abs(p1[0])<1e-10 && std::abs(p2[0])<1e-10)
//        {
//          return p1+w*(p2-p1);
//        }

      AssertThrow(std::abs(p1[0])<1e-10 && std::abs(p2[0])<1e-10,
                  ExcMessage("Other directions than alignment at x=0 currently not implemented."));
      const double a1 = std::atan2(p1[2]/cut_radius, p1[1]/main_radius);
      const double a2 = std::atan2(p2[2]/cut_radius, p2[1]/main_radius);
      const double angle = (1.0-w)*a1+w*a2;
      Point<dim> p;
      p[1] = main_radius*std::cos(angle);
      p[2] = cut_radius*std::sin(angle);
      return p;
    }
  private:
    const Point<dim> direction;
    const double main_radius;
    const double cut_radius;
  };

  template <int dim>
  class BdryFunc : public Function<dim>
  {
  public:
    BdryFunc (ProblemType::Kind problem)
      : Function<dim>(dim), problem(problem)
    {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

  private:
    ProblemType::Kind problem;
  };

  template <int dim>
  double BdryFunc<dim>::value (const Point<dim> &p,
                               const unsigned int component) const
  {
    if (problem==ProblemType::cylinder || problem==ProblemType::cylinder_shell)
      {
        if (abs(p(0)-1.0)<1e-5 && component==1)
          return -0.05;
        if (abs(p(0)-1.0)<1e-5 && component==0)
          return -0.1;
        if (abs(p(0)-1.0)<1e-5 && component==2)
          return 0.025;
      }

    if (problem==ProblemType::hyper_cube || problem==ProblemType::hyper_L)
      {
        if (abs(p(1)-1.0)<1e-5 && component==1)
          return -0.05;
        if (abs(p(1)-1.0)<1e-5 && component==0)
          return -0.1;
        if (abs(p(1)-1.0)<1e-5 && component==2)
          return 0.025;
      }

    if (problem==ProblemType::three_cylinders)
      {
        if (p(2)>0.5)
          {
            if (component == 0)
              return 0.3;
            if (component == 2)
              return -0.3;
            if (component == 1)
              return -0.1;
          }
      }

    return 0.0;
  }



  template <int dim>
  class MatrixIntegrator : public MeshWorker::LocalIntegrator<dim>
  {
  public:
    void cell(MeshWorker::DoFInfo<dim> &dinfo,
              typename MeshWorker::IntegrationInfo<dim> &info) const;
    void boundary(MeshWorker::DoFInfo<dim> &dinfo,
                  typename MeshWorker::IntegrationInfo<dim> &info) const;
    void face(MeshWorker::DoFInfo<dim> &dinfo1,
              MeshWorker::DoFInfo<dim> &dinfo2,
              typename MeshWorker::IntegrationInfo<dim> &info1,
              typename MeshWorker::IntegrationInfo<dim> &info2) const;

    double lambda, mu;

    MatrixIntegrator()
      : lambda (1.0), mu (1.0)
    {}
  };


  template <int dim>
  void MatrixIntegrator<dim>::cell(
    MeshWorker::DoFInfo<dim> &dinfo,
    typename MeshWorker::IntegrationInfo<dim> &info) const
  {
    dealii::LocalIntegrators::Elasticity::cell_matrix(
      dinfo.matrix(0, false).matrix, info.fe_values(0), 2. * mu);
    dealii::LocalIntegrators::GradDiv::cell_matrix(
      dinfo.matrix(0, false).matrix, info.fe_values(0), lambda);
  }


  template <int dim>
  void MatrixIntegrator<dim>::boundary(
    MeshWorker::DoFInfo<dim> &dinfo,
    typename MeshWorker::IntegrationInfo<dim> &info) const
  {
    if (dinfo.face->boundary_id()==1 || dinfo.face->boundary_id()==5)
      {

        const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
        LocalIntegrators::Elasticity::nitsche_matrix(
          dinfo.matrix(0,false).matrix, info.fe_values(0),
          penalty_pre_factor*LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
          2. * mu);


        LocalIntegrators::GradDiv::nitsche_matrix(
          dinfo.matrix(0,false).matrix, info.fe_values(0),
          penalty_pre_factor*LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
          lambda);
      }
  }

  template <int dim>
  void MatrixIntegrator<dim>::face(
    MeshWorker::DoFInfo<dim> &dinfo1,
    MeshWorker::DoFInfo<dim> &dinfo2,
    typename MeshWorker::IntegrationInfo<dim> &info1,
    typename MeshWorker::IntegrationInfo<dim> &info2) const
  {
    const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();

    LocalIntegrators::Elasticity::ip_matrix(
      dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix,
      dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
      info1.fe_values(0), info2.fe_values(0),
      penalty_pre_factor*LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
      2. * mu);
    LocalIntegrators::GradDiv::ip_matrix(
      dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix,
      dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
      info1.fe_values(0), info2.fe_values(0),
      penalty_pre_factor*LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
      lambda);
  }

  template <int dim>
  class RHSIntegrator : public MeshWorker::LocalIntegrator<dim>
  {
  public:

    RHSIntegrator(Function<dim> &boundary_values)
      : boundary_values(boundary_values)
    {}

    void cell(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
    void boundary(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
    void face(MeshWorker::DoFInfo<dim> &dinfo1,
              MeshWorker::DoFInfo<dim> &dinfo2,
              typename MeshWorker::IntegrationInfo<dim> &info1,
              typename MeshWorker::IntegrationInfo<dim> &info2) const;

  private:
    Function<dim> &boundary_values;
  };


  template <int dim>
  void RHSIntegrator<dim>::cell(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const
  {
    // * volume force term
    if (false)
      {
        const FEValuesBase<dim> &fe = info.fe_values();
        const unsigned int n_points = fe.n_quadrature_points;
        const unsigned int n_dofs = fe.dofs_per_cell;
        Vector<double> &b = dinfo.vector(0).block(0);

        for (unsigned int k=0; k<n_points; ++k)
          {
            const double fval[dim] = {1e-3, 0.0, 0.0};
            for (unsigned int i=0; i<n_dofs; ++i)
              {
                const unsigned int component_i =
                  info.finite_element().system_to_component_index(i).first;
                b(i) += fe.shape_value(i,k) *
                        fval[component_i] *
                        fe.JxW(k);
              }
          }
      }
  }


  template <int dim>
  void RHSIntegrator<dim>::boundary(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const
  {
    double mu = 1.0;

    if (dinfo.face->boundary_id()==1 || dinfo.face->boundary_id()==5)
      {
        const FEValuesBase<dim> &fe = info.fe_values();
        const unsigned int deg = fe.get_fe().tensor_degree();

        std::vector<std::vector<double> > input (dim, std::vector<double>(fe.n_quadrature_points));
        std::vector<std::vector<Tensor<1,dim> > > Dinput (dim, std::vector<Tensor<1,dim> >(fe.n_quadrature_points));
        std::vector<std::vector<double> > data (dim, std::vector<double>(fe.n_quadrature_points));

        for (int d=0; d<dim; ++d)
          boundary_values.value_list(fe.get_quadrature_points(), data[d], d);

        LocalIntegrators::Elasticity::nitsche_residual(
              dinfo.vector(0).block(0),
              fe,
              ArrayView<const std::vector<double>> (input),
              ArrayView<const std::vector<Tensor<1, dim>>> (Dinput),
              ArrayView<const std::vector<double>> (data),
              penalty_pre_factor*dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
              2. * mu
              );

        LocalIntegrators::GradDiv::nitsche_residual(
              dinfo.vector(0).block(0),
              fe,
              ArrayView<const std::vector<double>> (input),
              ArrayView<const std::vector<Tensor<1, dim>>> (Dinput),
              ArrayView<const std::vector<double>> (data),
              penalty_pre_factor*dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
              1.0
              );
      }
  }


  template <int dim>
  void RHSIntegrator<dim>::face(MeshWorker::DoFInfo<dim> &/*dinfo1*/, MeshWorker::DoFInfo<dim> &/*dinfo2*/,
                                MeshWorker::IntegrationInfo<dim> &/*info1*/, MeshWorker::IntegrationInfo<dim> &/*info2*/) const
  {
  }


  template <int dim>
  class Estimator : public MeshWorker::LocalIntegrator<dim>
  {
  public:
    void cell(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
    void boundary(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
    void face(MeshWorker::DoFInfo<dim> &dinfo1,
              MeshWorker::DoFInfo<dim> &dinfo2,
              typename MeshWorker::IntegrationInfo<dim> &info1,
              typename MeshWorker::IntegrationInfo<dim> &info2) const;
  };


  template <int dim>
  void Estimator<dim>::cell(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const
  {
    const FEValuesBase<dim> &fe = info.fe_values();

    const std::vector<Tensor<2,dim> > &DDuh = info.hessians[0][0];
    for (unsigned k=0; k<fe.n_quadrature_points; ++k)
      {
        const double t = dinfo.cell->diameter() * trace(DDuh[k]);
        dinfo.value(0) +=  t*t * fe.JxW(k);
      }
    dinfo.value(0) = std::sqrt(dinfo.value(0));
  }

  template <int dim>
  void Estimator<dim>::boundary(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const
  {
//    const FEValuesBase<dim> &fe = info.fe_values();

//    std::vector<double> boundary_values(fe.n_quadrature_points);
//    exact_solution.value_list(fe.get_quadrature_points(), boundary_values);

//    const std::vector<double> &uh = info.values[0][0];

//    const unsigned int deg = fe.get_fe().tensor_degree();
//    const double penalty = 2. * deg * (deg+1) * dinfo.face->measure() / dinfo.cell->measure();

//    for (unsigned k=0; k<fe.n_quadrature_points; ++k)
//      dinfo.value(0) += penalty * (boundary_values[k] - uh[k]) * (boundary_values[k] - uh[k])
//                        * fe.JxW(k);
//    dinfo.value(0) = std::sqrt(dinfo.value(0));
  }


  template <int dim>
  void Estimator<dim>::face(MeshWorker::DoFInfo<dim> &dinfo1,
                            MeshWorker::DoFInfo<dim> &dinfo2,
                            typename MeshWorker::IntegrationInfo<dim> &info1,
                            typename MeshWorker::IntegrationInfo<dim> &info2) const
  {
    const FEValuesBase<dim> &fe = info1.fe_values();
    const std::vector<double> &uh1 = info1.values[0][0];
    const std::vector<double> &uh2 = info2.values[0][0];
    const std::vector<Tensor<1,dim> > &Duh1 = info1.gradients[0][0];
    const std::vector<Tensor<1,dim> > &Duh2 = info2.gradients[0][0];

    const unsigned int deg = fe.get_fe().tensor_degree();
    const double penalty1 = deg * (deg+1) * dinfo1.face->measure() / dinfo1.cell->measure();
    const double penalty2 = deg * (deg+1) * dinfo2.face->measure() / dinfo2.cell->measure();
    const double penalty = penalty1 + penalty2;
    const double h = dinfo1.face->measure();

    for (unsigned k=0; k<fe.n_quadrature_points; ++k)
      {
        double diff1 = uh1[k] - uh2[k];
        double diff2 = fe.normal_vector(k) * Duh1[k] - fe.normal_vector(k) * Duh2[k];
        dinfo1.value(0) += (penalty * diff1*diff1 + h * diff2*diff2)
                           * fe.JxW(k);
      }
    dinfo1.value(0) = std::sqrt(dinfo1.value(0));
    dinfo2.value(0) = dinfo1.value(0);
    // do not fill values if cells are ghost cells because we don't communicate
    if (!dinfo1.cell->is_locally_owned())
      dinfo1.value(0) = 0.0;
    if (!dinfo2.cell->is_locally_owned())
      dinfo2.value(0) = 0.0;
  }





  template <int dim>
  class InteriorPenaltyProblem
  {
  public:
    typedef MeshWorker::IntegrationInfo<dim> CellInfo;

    InteriorPenaltyProblem(const std::string& prm_filename);

    void run(unsigned int n_steps);

  private:
    void setup_system ();
    void assemble_matrix ();
    void assemble_mg_matrix ();
    void assemble_right_hand_side ();
    void error ();
    double estimate ();
    void solve ();
    void output_results (const unsigned int cycle);

    struct Settings
    {
      Settings(const std::string &prm_filename);
      ProblemType::Kind problem;
      bool use_amg;

      void parse(const std::string &prm_filename);
    };
    Settings settings;

    ConditionalOStream                        pcout;
    parallel::distributed::Triangulation<dim>        triangulation;
    const MappingQGeneric<dim> mapping;
    const FESystem<dim> fe;
    DoFHandler<dim>           dof_handler;



    IndexSet locally_relevant_set;

    LA::MPI::SparseMatrix matrix;
    LA::MPI::Vector       solution;
    LA::MPI::Vector       right_hand_side;
    BlockVector<double>  estimates;

    MGLevelObject<LA::MPI::SparseMatrix> mg_matrix;

    MGLevelObject<LA::MPI::SparseMatrix> mg_matrix_dg_down;
    MGLevelObject<LA::MPI::SparseMatrix> mg_matrix_dg_up;

    TimerOutput computing_timer;


  };

  template <int dim>
  InteriorPenaltyProblem<dim>::Settings::Settings(const std::string &prm_filename)
  {
    parse(prm_filename);
  }


  template <int dim>
  void
  InteriorPenaltyProblem<dim>::Settings::parse(const std::string &prm_filename)
  {
    ParameterHandler prm;
    prm.declare_entry("test problem", "hyper_L",
                      Patterns::Selection(ProblemType::get_options()),
                      "Select problem to solve, options:" + ProblemType::get_options());
    prm.declare_entry("use AMG only", "false",
                      Patterns::Bool(),
                      "");

    try
      {
        prm.parse_input(prm_filename);
      }
    catch(...)
      {
        prm.print_parameters(std::cout, ParameterHandler::Text);
        std::exit(-1);
      }
    this->problem = ProblemType::parse(prm.get("test problem"));
    this->use_amg = prm.get_bool("use AMG only");
  }

  template <int dim>
  InteriorPenaltyProblem<dim>::InteriorPenaltyProblem(const std::string &prm_filename)
    :
    settings(prm_filename),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            == 0)),
    triangulation (MPI_COMM_WORLD,Triangulation<dim>::
                   limit_level_difference_at_vertices,
                   parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    mapping(2),
    fe(FE_DGQ<dim>(1),dim),
    dof_handler(triangulation),
    estimates(1),
    computing_timer (pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)
  {
    if (settings.use_amg)
      pcout << "using amg!" << std::endl;

    if (settings.problem==ProblemType::hyper_L
        || settings.problem==ProblemType::hyper_cube)
      {
        if (settings.problem==ProblemType::hyper_L)
        GridGenerator::hyper_L(triangulation, -1, 1, false);
        else
          GridGenerator::hyper_cube(triangulation, -1, 1, false);

        typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (; cell!=endc; ++cell)
          for (unsigned int f=0;
               f < GeometryInfo<dim>::faces_per_cell;
               ++f)
            {
              const Point<dim> face_center = cell->face(f)->center();
              if (cell->face(f)->at_boundary())
                {
                  if ( abs(face_center[1]+1.0) < 1e-5)
                    cell->face(f)->set_boundary_id(1);
                  if ( abs(face_center[1]-1.0) < 1e-5)
                    cell->face(f)->set_boundary_id(5);
                }
            }
        triangulation.refine_global(1);
      }

    if (settings.problem==ProblemType::cylinder ||
        settings.problem==ProblemType::cylinder_shell)
      {
        if (settings.problem==ProblemType::cylinder)
          {
            GridGenerator::cylinder (triangulation, 1.0, 1.0);
            triangulation.set_all_manifold_ids(2);
          }
        else
          {
            // shift and rotate to be aligned like the cylinder
            GridGenerator::cylinder_shell (triangulation, 2.0, 0.1, 0.3, 0, 0);
            Tensor<1,dim> shift;
            shift[2] = -1.0;
            GridTools::shift (shift, triangulation);
            GridTools::rotate (numbers::PI/2.0, 1, triangulation);
            triangulation.set_all_manifold_ids(1);
          }

        for (typename Triangulation<dim,dim>::active_cell_iterator cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
          {
            for (unsigned int f=0;
                 f < GeometryInfo<dim>::faces_per_cell;
                 ++f)
              {
                const Point<dim> face_center = cell->face(f)->center();
                if (cell->face(f)->at_boundary())
                  {
                    if ( abs(face_center[0]+1.0) < 1e-5)
                      cell->face(f)->set_boundary_id(1);
                    if ( abs(face_center[0]-1.0) < 1e-5)
                      cell->face(f)->set_boundary_id(5);

                    if (settings.problem==ProblemType::cylinder)
                      {
                        bool curved_part = false;
                        for (unsigned int i=1; i<GeometryInfo<dim>::vertices_per_face; ++i)
                          if (std::abs(cell->face(f)->vertex(i)[0] - cell->face(f)->vertex(0)[0]) > 1e-10)
                            curved_part = true;
                        if (curved_part)
                          cell->face(f)->set_all_manifold_ids(1);
                      }
                  }
              }
          }
        static CylindricalManifold<dim> cylinder(0);
        triangulation.set_manifold(1, cylinder);
        static TransfiniteInterpolationManifold<dim> transfinite;
        transfinite.initialize(triangulation);
        triangulation.set_manifold(2, transfinite);
        triangulation.refine_global(1);
      }

    if (settings.problem==ProblemType::three_cylinders)
      {
        GridIn<dim> grid_in;
        std::ifstream input_3D("threecylinders.dat");
        grid_in.attach_triangulation (triangulation);
        grid_in.read_ucd(input_3D);

        Point<dim> s1, s2, s3;
        s1[dim-1] = 1;
        s2[0] = sqrt(0.5);
        s2[dim-1] = sqrt(0.5);
        s3[0] = -sqrt(0.5);
        s3[dim-1] = sqrt(0.5);

        triangulation.set_all_manifold_ids(0);

        // set manifold ids of boundaries
        for (typename Triangulation<dim>::active_cell_iterator cell=
               triangulation.begin_active(); cell != triangulation.end(); ++cell)
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary())
              {
                // check whether all vertices of the face are at a given
                // radius from the origin, i.e., the cylinder boundaries
                //const double radius = 2.;
                Tensor<1,dim> d1 = cell->face(f)->vertex(1)-cell->face(f)->vertex(0);
                Tensor<1,dim> d2 = cell->face(f)->vertex(2)-cell->face(f)->vertex(0);
                Tensor<1,dim> normal = cross_product_3d(d1, d2);
                normal /= normal.norm();
                //std::cout << normal << "   ";
                if (std::abs(normal[dim-1]+1) < 1e-12)
                  cell->face(f)->set_boundary_id(1);
                if ((std::abs(normal[0]+std::sqrt(0.5)) < 1e-12 &&
                     std::abs(normal[dim-1]+std::sqrt(0.5)) < 1e-12))
                  cell->face(f)->set_boundary_id(5);

                // exclude circular cylinder faces
                if (std::abs(normal[dim-1]+1) < 1e-12 ||
                    (std::abs(normal[0]+std::sqrt(0.5)) < 1e-12 &&
                     std::abs(normal[dim-1]+std::sqrt(0.5)) < 1e-12) ||
                    (std::abs(normal[0]-std::sqrt(0.5)) < 1e-12 &&
                     std::abs(normal[dim-1]+std::sqrt(0.5)) < 1e-12))
                  continue;
                //std::cout << "detected: " << cell->face(f)->center() << " " << normal << std::endl;

                // detect axis along which we are aligned by identifying
                // two parallel lines along the surface. The problem is
                // that we do not know the orientation so we have to try
                // the different directions out. We do this by first
                // finding the dominant direction (the longer of the the
                // faces) and then check what cylinder direction the
                // current face is most parallel to. This is a hack but
                // it works on the given geometry.
                Tensor<1,dim> r1 = cell->face(f)->vertex(1)-cell->face(f)->vertex(0);
                //Tensor<1,dim> r2 = cell->face(f)->vertex(3)-cell->face(f)->vertex(2);
                Tensor<1,dim> r3 = cell->face(f)->vertex(2)-cell->face(f)->vertex(0);
                //Tensor<1,dim> r4 = cell->face(f)->vertex(3)-cell->face(f)->vertex(1);
                if (r1.norm() > r3.norm())
                  {
                    double l1 = std::abs(r1*s1), l2 = std::abs(r1*s2), l3 = std::abs(r1*s3);
                    if (l1 > l2 && l1 > l3)
                      cell->face(f)->set_all_manifold_ids(1);
                    else if (l2 > l1 && l2 > l3)
                      cell->face(f)->set_all_manifold_ids(2);
                    else
                      cell->face(f)->set_all_manifold_ids(3);
                  }
                else
                  {
                    double l1 = std::abs(r3*s1), l2 = std::abs(r3*s2), l3 = std::abs(r3*s3);
                    if (l1 > l2 && l1 > l3)
                      cell->face(f)->set_all_manifold_ids(1);
                    else if (l2 > l1 && l2 > l3)
                      cell->face(f)->set_all_manifold_ids(2);
                    else
                      cell->face(f)->set_all_manifold_ids(3);
                  }
                for (unsigned int l=0; l<GeometryInfo<dim>::lines_per_face; ++l)
                  if (std::abs(cell->face(f)->line(l)->center()[0]) < 1e-10 &&
                      cell->face(f)->line(l)->center()[2] > 0)
                    cell->face(f)->line(l)->set_manifold_id(5);
                //Tensor<1,dim> c1 = cross_product_3d(r1, r2);
                //Tensor<1,dim> c2 = cross_product_3d(r3, r4);
                //if (c1.norm() < 1e-10)
                //  std::cout << "v1: " << r1 << "   " << r2 << std::endl;
                //else if (c2.norm() < 1e-10)
                //  std::cout << "v2: " << r3 << "   " << r4 << std::endl;
                //else
                //  std::cout << "v3: " << r1 << "   " << r2 << "   " << c1 << "       " << r3 << "   " << r4 << "   " << c2 << std::endl;
              }
        static CylindricalManifold<dim> cylinder1(s1, Point<dim>());
        triangulation.set_manifold(1, cylinder1);
        static CylindricalManifold<dim> cylinder2(s2, Point<dim>());
        triangulation.set_manifold(2, cylinder2);
        static CylindricalManifold<dim> cylinder3(s3, Point<dim>());
        triangulation.set_manifold(3, cylinder3);
        static EllipseManifold<dim> ellipse1(s2+s3, 2., std::sqrt(8));
        triangulation.set_manifold(5, ellipse1);

        static TransfiniteInterpolationManifold<dim> transfinite;
        transfinite.initialize(triangulation);
        triangulation.set_manifold(0, transfinite);

        triangulation.refine_global(1);
      }

  }


  template <int dim>
  void
  InteriorPenaltyProblem<dim>::setup_system()
  {
    {
      TimerOutput::Scope timing (computing_timer, "Setup");
      dof_handler.distribute_dofs(fe);

      DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_set);
      solution.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
      right_hand_side.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);

      DynamicSparsityPattern c_sparsity(dof_handler.n_dofs(), dof_handler.n_dofs());
      DoFTools::make_flux_sparsity_pattern(dof_handler, c_sparsity);
      matrix.reinit(dof_handler.locally_owned_dofs(), c_sparsity, MPI_COMM_WORLD, true);
    }

    if (!settings.use_amg)
      {

        TimerOutput::Scope timing2 (computing_timer, "Setup MG");
        dof_handler.distribute_mg_dofs ();

        const unsigned int n_levels = triangulation.n_global_levels();
        mg_matrix.resize(0, n_levels-1);
        mg_matrix.clear_elements();
        mg_matrix_dg_up.resize(0, n_levels-1);
        mg_matrix_dg_up.clear_elements();
        mg_matrix_dg_down.resize(0, n_levels-1);
        mg_matrix_dg_down.clear_elements();

        for (unsigned int level=mg_matrix.min_level();
             level<=mg_matrix.max_level(); ++level)
          {
            DynamicSparsityPattern c_sparsity(dof_handler.n_dofs(level));
            MGTools::make_flux_sparsity_pattern(dof_handler, c_sparsity, level);
            mg_matrix[level].reinit(dof_handler.locally_owned_mg_dofs(level),
                                    dof_handler.locally_owned_mg_dofs(level),
                                    c_sparsity,
                                    MPI_COMM_WORLD, true);

            if (level>0)
              {
                DynamicSparsityPattern ci_sparsity;
                ci_sparsity.reinit(dof_handler.n_dofs(level-1), dof_handler.n_dofs(level));
                MGTools::make_flux_sparsity_pattern_edge(dof_handler, ci_sparsity, level);

                mg_matrix_dg_up[level].reinit(dof_handler.locally_owned_mg_dofs(level-1),
                                              dof_handler.locally_owned_mg_dofs(level),
                                              ci_sparsity,
                                              MPI_COMM_WORLD, true);
                mg_matrix_dg_down[level].reinit(dof_handler.locally_owned_mg_dofs(level-1),
                                                dof_handler.locally_owned_mg_dofs(level),
                                                ci_sparsity,
                                                MPI_COMM_WORLD, true);
              }
          }
      }
  }


  template <int dim>
  void
  InteriorPenaltyProblem<dim>::assemble_matrix()
  {
    TimerOutput::Scope timing (computing_timer, "Assemble");
    MeshWorker::IntegrationInfoBox<dim> info_box;
    UpdateFlags update_flags = update_values | update_gradients;
    info_box.add_update_flags_all(update_flags);
    info_box.initialize(fe, mapping);

    MeshWorker::DoFInfo<dim> dof_info(dof_handler);

    MeshWorker::Assembler::MatrixSimple<LA::MPI::SparseMatrix> assembler;
    assembler.initialize(matrix);

    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    begin(IteratorFilters::LocallyOwnedCell(), dof_handler.begin_active());
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    end(IteratorFilters::LocallyOwnedCell(), dof_handler.end());

    MatrixIntegrator<dim> integrator;
    MeshWorker::integration_loop<dim, dim>(
      begin, end,
      dof_info, info_box,
      integrator, assembler);

    matrix.compress(VectorOperation::add);
  }


  template <int dim>
  void
  InteriorPenaltyProblem<dim>::assemble_mg_matrix()
  {
    TimerOutput::Scope timing (computing_timer, "Assemble MG");
    MeshWorker::IntegrationInfoBox<dim> info_box;
    UpdateFlags update_flags = update_values | update_gradients;
    info_box.add_update_flags_all(update_flags);
    info_box.initialize(fe, mapping);

    MeshWorker::DoFInfo<dim> dof_info(dof_handler);

    MeshWorker::Assembler::MGMatrixSimple<LA::MPI::SparseMatrix> assembler;
    assembler.initialize(mg_matrix);
    assembler.initialize_fluxes(mg_matrix_dg_up, mg_matrix_dg_down);

    FilteredIterator<typename DoFHandler<dim>::level_cell_iterator>
    begin(IteratorFilters::LocallyOwnedLevelCell(), dof_handler.begin());
    FilteredIterator<typename DoFHandler<dim>::level_cell_iterator>
    end(IteratorFilters::LocallyOwnedLevelCell(), dof_handler.end());

    MatrixIntegrator<dim> integrator;
    MeshWorker::integration_loop<dim, dim> (
      begin, end,
      dof_info, info_box,
      integrator, assembler);

    for (unsigned int level=mg_matrix.min_level(); level <= mg_matrix.max_level(); ++level)
      {
        mg_matrix[level].compress(VectorOperation::add);
        if (level > mg_matrix.min_level())
          {
            mg_matrix_dg_up[level].compress(VectorOperation::add);
            mg_matrix_dg_down[level].compress(VectorOperation::add);
          }
      }
  }


  template <int dim>
  void
  InteriorPenaltyProblem<dim>::assemble_right_hand_side()
  {
    TimerOutput::Scope timing (computing_timer, "Assemble RHS");
    MeshWorker::IntegrationInfoBox<dim> info_box;
    UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
    info_box.add_update_flags_all(update_flags);
    info_box.initialize(fe, mapping);

    MeshWorker::DoFInfo<dim> dof_info(dof_handler);

    MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector> assembler;
    AnyData data;
    data.add<LA::MPI::Vector *>(&right_hand_side, "RHS");
    assembler.initialize(data);

    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    begin(IteratorFilters::LocallyOwnedCell(), dof_handler.begin_active());
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    end(IteratorFilters::LocallyOwnedCell(), dof_handler.end());


    BdryFunc<dim> boundary_values(settings.problem);
    RHSIntegrator<dim> integrator (boundary_values);
    MeshWorker::integration_loop<dim, dim>(begin, end,
                                           dof_info, info_box,
                                           integrator, assembler);

    right_hand_side.compress(VectorOperation::add);
    right_hand_side *= -1.;
  }

  template<typename VECTOR>
  class MGCoarseAMG : public MGCoarseGridBase<VECTOR>
  {
  public:
    MGCoarseAMG(const LA::MPI::SparseMatrix &coarse_matrix,
                const LA::MPI::PreconditionAMG::AdditionalData additional_data)
    {
      precondition_amg.initialize(coarse_matrix, additional_data);
    }

    virtual void operator() (const unsigned int,
                             VECTOR &dst,
                             const VECTOR &src) const
    {
      precondition_amg.vmult(dst, src);
    }

  private:
    LA::MPI::PreconditionAMG precondition_amg;
  };



  template <int dim>
  void
  InteriorPenaltyProblem<dim>::solve()
  {
    TimerOutput::Scope timing (computing_timer, "Solve");
    SolverControl control(5000, 1.e-8*right_hand_side.l2_norm(), true);
    SolverCG<LA::MPI::Vector > solver(control);

    if (settings.use_amg)
      {
        TrilinosWrappers::PreconditionAMG prec;

        {
          TimerOutput::Scope timing (computing_timer, "setup AMG");

          const bool extended_nullspace = true;
#ifdef DEAL_II_WITH_TRILINOS
          if (extended_nullspace)
            {
              const int nullspace_dimension = dim==3 ? 6 : 3;
              const IndexSet &locally_owned = dof_handler.locally_owned_dofs();
              std::vector<double> nullspace(nullspace_dimension*locally_owned.n_elements());
              std::vector<std::vector<double> > coordinates(dim,
                                                            std::vector<double>(locally_owned.n_elements()));
              Quadrature<dim> support_quadrature(fe.get_unit_support_points());
              FEValues<dim> fe_values(mapping, fe, support_quadrature,
                                      update_quadrature_points);
              std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
              for (typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
                   cell != dof_handler.end(); ++cell)
                if (cell->is_locally_owned())
                  {
                    fe_values.reinit(cell);
                    cell->get_dof_indices(dof_indices);
                    for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
                      if (locally_owned.is_element(dof_indices[i]))
                        {
                          const unsigned int index = locally_owned.index_within_set(dof_indices[i]);
                          const unsigned int c=fe.system_to_component_index(i).first;
                          const std::size_t offset = locally_owned.n_elements();
                          const Point<dim> p = fe_values.quadrature_point(i);
                          // use rigid body rotation around origin, might want to
                          // adjust this
                          const Point<dim> origin;
                          if (dim == 2)
                            {
                              switch (c)
                                {
                                case 0:
                                  nullspace[0*offset+index] = 1.0;
                                  nullspace[1*offset+index] = 0.0;
                                  nullspace[2*offset+index] = -p[1] + origin[1];
                                  break;
                                case 1:
                                  nullspace[0*offset+index] = 0.0;
                                  nullspace[1*offset+index] = 1.0;
                                  nullspace[2*offset+index] = p[0] - origin[0];
                                  break;
                                }
                            }
                          else if (dim == 3)
                            {
                              switch (c)
                                {
                                case 0:
                                  nullspace[0*offset+index] = 1.0;
                                  nullspace[1*offset+index] = 0.0;
                                  nullspace[2*offset+index] = 0.0;
                                  nullspace[3*offset+index] = 0.0;
                                  nullspace[4*offset+index] = p[2] - origin[2];
                                  nullspace[5*offset+index] = -p[1] + origin[1];
                                  break;
                                case 1:
                                  nullspace[0*offset+index] = 0.0;
                                  nullspace[1*offset+index] = 1.0;
                                  nullspace[2*offset+index] = 0.0;
                                  nullspace[3*offset+index] = -p[2] + origin[2];
                                  nullspace[4*offset+index] = 0.0;
                                  nullspace[5*offset+index] = p[0] - origin[0];
                                  break;
                                case 2:
                                  nullspace[0*offset+index] = 0.0;
                                  nullspace[1*offset+index] = 0.0;
                                  nullspace[2*offset+index] = 1.0;
                                  nullspace[3*offset+index] = 0.0;
                                  nullspace[4*offset+index] = p[1] - origin[1];
                                  nullspace[5*offset+index] = -p[0] + origin[0];
                                  break;
                                }
                            }
                          for (unsigned int d=0; d<dim; ++d)
                            coordinates[d][index] = p[d];
                        }
                  }

              Teuchos::ParameterList parameter_list;
              ML_Epetra::SetDefaults("SA",parameter_list);
              parameter_list.set("smoother: type", "Chebyshev");
              parameter_list.set("smoother: Chebyshev alpha", 20.);
              //ML_Epetra::SetDefaults("NSSA",parameter_list);
              //parameter_list.set("aggregation: block scaling", true);
              parameter_list.set("aggregation: threshold", 0.02);
              parameter_list.set("aggregation: type", "Uncoupled");
              parameter_list.set("smoother: sweeps", 4);
              parameter_list.set("coarse: max size", 2000);

              parameter_list.set("repartition: enable",1);
              parameter_list.set("repartition: max min ratio",1.3);
              parameter_list.set("repartition: min per proc",300);
              parameter_list.set("repartition: partitioner","Zoltan");
              parameter_list.set("repartition: Zoltan dimensions", dim);
              if (coordinates.size() > 0 && coordinates[0].size() > 0)
                {
                  parameter_list.set("x-coordinates", const_cast<double *>(&coordinates[0][0]));
                  parameter_list.set("y-coordinates", const_cast<double *>(&coordinates[1][0]));
                  if (coordinates.size() > 2)
                    parameter_list.set("z-coordinates", const_cast<double *>(&coordinates[2][0]));
                }

              parameter_list.set("ML output", 10);

              parameter_list.set("null space: type", "pre-computed");
              parameter_list.set("null space: dimension",
                                 nullspace_dimension);
              parameter_list.set("null space: vectors",
                                 &nullspace[0]);

              prec.initialize (matrix, parameter_list);
            }
          else
#endif
            {
              TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
              DoFTools::extract_constant_modes(dof_handler, std::vector<bool>(dim,true),
                                               Amg_data.constant_modes);
              Amg_data.elliptic = true;
              Amg_data.higher_order_elements = false;
              Amg_data.smoother_sweeps = 4;
              Amg_data.aggregation_threshold = 0.02;
              Amg_data.output_details = true;

              prec.initialize (matrix,
                               Amg_data);
            }
        }

        solver.solve (matrix, solution, right_hand_side,
                      prec);

        pcout << "    CG converged in " << control.last_step() << " iterations." << std::endl;
      }
    else
      {
        MGTransferPrebuilt<LA::MPI::Vector > mg_transfer;
        mg_transfer.build_matrices(dof_handler);

        ReductionControl coarse_solver_control (1000, 1e-14, 1e-5, false, true);
        SolverCG<LA::MPI::Vector> coarse_solver(coarse_solver_control);

        LA::MPI::SparseMatrix &coarse_matrix = mg_matrix[0];

        LA::MPI::PreconditionAMG::AdditionalData Amg_data;
        Amg_data.elliptic = true;
        Amg_data.higher_order_elements = true;
        Amg_data.n_cycles = 1;
        Amg_data.smoother_sweeps = 2;
        Amg_data.aggregation_threshold = 0.1;
        Amg_data.smoother_type = "symmetric Gauss-Seidel";
        const IndexSet &index_set = dof_handler.locally_owned_mg_dofs(mg_matrix.min_level());
        Amg_data.constant_modes.resize(dim, std::vector<bool>(index_set.n_elements()));
        std::vector<types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);
        for (typename DoFHandler<dim>::cell_iterator cell=dof_handler.begin(mg_matrix.min_level());
             cell != dof_handler.end(mg_matrix.min_level()); ++cell)
          if (cell->level_subdomain_id() == triangulation.locally_owned_subdomain())
            {
              cell->get_mg_dof_indices(dof_indices);
              for (unsigned int i=0; i<dof_indices.size(); ++i)
                if (index_set.is_element(dof_indices[i]))
                  Amg_data.constant_modes[dof_handler.get_fe().system_to_component_index(i).first][index_set.index_within_set(dof_indices[i])] = true;
            }

        //Amg_data.output_details = true;

        LA::MPI::PreconditionAMG prec;
        prec.initialize(coarse_matrix, Amg_data);

        //MGCoarseAMG<LA::MPI::Vector> coarse_grid_solver(coarse_matrix, Amg_data);


        MGCoarseGridIterativeSolver<LA::MPI::Vector,
                                    SolverCG<LA::MPI::Vector>,
                                    LA::MPI::SparseMatrix,
                                    LA::MPI::PreconditionAMG> coarse_grid_solver(coarse_solver,
                                        coarse_matrix,
                                        prec);

        typedef TrilinosWrappers::PreconditionJacobi Smoother;
        //typedef TrilinosWrappers::PreconditionILU Smoother;
        MGSmootherPrecondition<LA::MPI::SparseMatrix, Smoother, LA::MPI::Vector> mg_smoother;
        mg_smoother.initialize(mg_matrix, (dim==2)?0.7:.5);
        mg_smoother.set_steps(5);

        mg::Matrix<LA::MPI::Vector> mgmatrix(mg_matrix);
        mg::Matrix<LA::MPI::Vector> mgdown(mg_matrix_dg_down);
        mg::Matrix<LA::MPI::Vector> mgup(mg_matrix_dg_up);

        Multigrid<LA::MPI::Vector > mg(mgmatrix,
                                       coarse_grid_solver, mg_transfer,
                                       mg_smoother, mg_smoother);
        mg.set_edge_flux_matrices(mgdown, mgup);

        PreconditionMG<dim, LA::MPI::Vector,
                       MGTransferPrebuilt<LA::MPI::Vector > >
                       preconditioner(dof_handler, mg, mg_transfer);
        solver.solve(matrix, solution, right_hand_side, preconditioner);
        pcout << "    CG converged in " << control.last_step() << " iterations." << std::endl;
        pcout << "    inner (last): " << coarse_solver_control.last_step() << std::endl;
      }

  }


  template <int dim>
  double
  InteriorPenaltyProblem<dim>::estimate()
  {
    TimerOutput::Scope timing (computing_timer, "Estimate");

    LA::MPI::Vector ghost;
    ghost.reinit(locally_relevant_set, MPI_COMM_WORLD);
    ghost = solution;

    std::vector<unsigned int> old_user_indices;
    triangulation.save_user_indices(old_user_indices);

    estimates.block(0).reinit(triangulation.n_active_cells());
    unsigned int i=0;
    for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell,++i)
      cell->set_user_index(i);

    MeshWorker::IntegrationInfoBox<dim> info_box;
    const unsigned int n_gauss_points = dof_handler.get_fe().tensor_degree()+1;
    info_box.initialize_gauss_quadrature(n_gauss_points, n_gauss_points+1, n_gauss_points);
    AnyData data;
    data.add<LA::MPI::Vector *>(&ghost, "solution");

    info_box.cell_selector.add("solution", false, false, true);
    info_box.boundary_selector.add("solution", true, true, false);
    info_box.face_selector.add("solution", true, true, false);

    info_box.add_update_flags_boundary(update_quadrature_points);
    info_box.initialize(fe, mapping, data, solution);

    MeshWorker::DoFInfo<dim> dof_info(dof_handler);

    MeshWorker::Assembler::CellsAndFaces<double> assembler;
    AnyData out_data;
    BlockVector<double> *est = &estimates;
    out_data.add(est, "cells");
    assembler.initialize(out_data, false);

    Estimator<dim> integrator;
    MeshWorker::LoopControl lctrl;
    // assemble all faces adjacent to ghost cells to get the full
    // information for all own cells without communication
    lctrl.faces_to_ghost = MeshWorker::LoopControl::both;

    MeshWorker::integration_loop<dim, dim> (
      dof_handler.begin_active(), dof_handler.end(),
      dof_info, info_box,
      integrator, assembler, lctrl);

    triangulation.load_user_indices(old_user_indices);
    // estimates is a BlockVector<double> (so serial) on each processor
    // with one entry per active cell. Note that only the locally owned
    // cells are !=0, so summing the contributions of l2_norm() over all
    // processors is the right way to do this.
    double local_norm = estimates.block(0).l2_norm();
    local_norm *= local_norm;
    return std::sqrt(Utilities::MPI::sum(local_norm, MPI_COMM_WORLD));
  }


  template<int dim>
  class LvlDataOut : public DataOut<dim>
  {
  public:
    LvlDataOut (const unsigned int subdomain_id, unsigned int lvl)
      :
      subdomain_id (subdomain_id), lvl(lvl)
    {}

    virtual typename DataOut<dim>::cell_iterator
    first_cell ()
    {
      typename DataOut<dim>::cell_iterator
      cell = this->dofs->begin(lvl);
      while ((cell != this->dofs->end(lvl)) &&
             (cell->level_subdomain_id() != subdomain_id))
        ++cell;

      if (cell == this->dofs->end(lvl))
        return this->dofs->end();

      return cell;
    }

    virtual typename DataOut<dim>::cell_iterator
    next_cell (const typename DataOut<dim>::cell_iterator &old_cell)
    {
      if (old_cell != this->dofs->end(lvl))
        {
          typename DataOut<dim>::cell_iterator
          cell = old_cell;
          ++cell;
          while ((cell != this->dofs->end(lvl)) &&
                 (cell->level_subdomain_id() != subdomain_id))
            ++cell;
          if (cell == this->dofs->end(lvl))
            return this->dofs->end();

          return cell;
        }
      else
        return this->dofs->end();
    }

    virtual typename DataOut<dim>::cell_iterator
    first_locally_owned_cell ()
    {
      return first_cell();
    }


    virtual typename DataOut<dim>::cell_iterator
    next_locally_owned_cell (const typename DataOut<dim>::cell_iterator &old_cell)
    {
      return next_cell(old_cell);
    }




  private:
    const unsigned int subdomain_id;
    const unsigned int lvl;
  };

  template<int dim>
  void output(parallel::distributed::Triangulation<dim> &tr)
  {
    const std::string filename = ("mesh." +
                                  Utilities::int_to_string
                                  (tr.locally_owned_subdomain(), 4) +
                                  ".svg");
    std::ofstream stream(filename.c_str());
    GridOut grid_out;
    GridOutFlags::Svg svg_flags;
    svg_flags.coloring = GridOutFlags::Svg::level_subdomain_id;
    svg_flags.label_material_id = false;
    svg_flags.label_level_number = false;
    svg_flags.label_cell_index = false;
    svg_flags.label_subdomain_id = true;
    svg_flags.label_level_subdomain_id = false;
    svg_flags.background = GridOutFlags::Svg::transparent;
    svg_flags.polar_angle = 60;
    svg_flags.convert_level_number_to_height = true;
    grid_out.set_flags(svg_flags);
    grid_out.write_svg(tr, stream);

    {
      const std::string filename = ("mesh." +
                                    Utilities::int_to_string
                                    (tr.locally_owned_subdomain(), 4) +
                                    ".fig");
      std::ofstream stream(filename.c_str());
      GridOutFlags::XFig flags;
      flags.color_by = GridOutFlags::XFig::level_subdomain_id;


      grid_out.write_xfig(tr, stream);


    }

  }

  template<int dim, class IT>
  void output_vtu_grid(const IT &begin,const IT &endc, std::ostream &out, bool exclude_invalid_lvl, bool exclude_invalid_active)
  {
    std::vector<DataOutBase::Patch<dim> > patches;
    unsigned int n_datasets=2;
    std::vector<std::string> data_names;
    data_names.push_back("subdomain");
    data_names.push_back("lvlsubdomain");

    unsigned int n_q_points=4;

    for (IT cell=begin; cell !=endc; ++cell)
      {
        if (exclude_invalid_lvl && cell->level_subdomain_id() == numbers::artificial_subdomain_id)
          continue;

        if (exclude_invalid_active && !cell->has_children() && cell->subdomain_id() == numbers::artificial_subdomain_id)
          continue;

        DataOutBase::Patch<dim> patch;

        for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_cell; ++vertex)
          patch.vertices[vertex] = cell->vertex(vertex);

        patch.data.reinit(n_datasets, n_q_points);
        patch.points_are_available = false;
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          patch.neighbors[f] = numbers::invalid_unsigned_int;

        for (unsigned int q=0; q<n_q_points; ++q)
          {
            patch.data(0,q) = cell->has_children()? -1.0 : (double)static_cast<int>(cell->subdomain_id());
            patch.data(1,q) = (double)static_cast<int>(cell->level_subdomain_id());
          }

        patches.push_back(patch);
      }



    std::vector<std::tuple<unsigned int, unsigned int, std::string> > vector_data_ranges;
    DataOutBase::VtkFlags flags;
    DataOutBase::write_vtu (patches,
                            data_names,
                            vector_data_ranges,
                            flags,
                            out);



  }



  template<int dim>
  void output_vtu_grid(parallel::distributed::Triangulation<dim> &tr)
  {
    for (unsigned int lvl = 0; lvl < tr.n_global_levels(); ++lvl)
      {
        {
          const std::string filename = ("localmesh.lvl" +
                                        Utilities::int_to_string(lvl,2) + "." +
                                        Utilities::int_to_string(tr.locally_owned_subdomain(), 4) +
                                        ".vtu");
          std::ofstream out(filename.c_str());
          output_vtu_grid<dim>(tr.begin(lvl), tr.end(lvl),out, false, false);
        }

        {
          const std::string filename = ("mesh.lvl" +
                                        Utilities::int_to_string(lvl,2) + "." +
                                        Utilities::int_to_string(tr.locally_owned_subdomain(), 4) +
                                        ".vtu");
          std::ofstream out(filename.c_str());
          output_vtu_grid<dim>(tr.begin(lvl), tr.end(lvl),out, true, false);
        }


      }
    const std::string filename = ("localmesh." +
                                  Utilities::int_to_string
                                  (tr.locally_owned_subdomain(), 4) +

                                  ".vtu");
    std::ofstream out(filename.c_str());
    output_vtu_grid<dim,typename Triangulation<dim>::active_cell_iterator >
    (tr.begin_active(), tr.end(), out, false, false);
  }


  template <int dim>
  void InteriorPenaltyProblem<dim>::output_results (const unsigned int cycle)
  {
    TimerOutput::Scope timing (computing_timer, "Data Out");
    //output(triangulation);
    //output_vtu_grid(triangulation);


    DataOut<dim> data_out;

    LA::MPI::Vector temp_solution;
    temp_solution.reinit(locally_relevant_set, MPI_COMM_WORLD);
    temp_solution = solution;

    data_out.attach_dof_handler (dof_handler);
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> names(dim, "displacement");

    data_out.add_data_vector (dof_handler, temp_solution, names, data_component_interpretation);
    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches (0);

    const std::string filename = ("solution-" +
                                  Utilities::int_to_string (cycle, 5) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4) +
                                  ".vtu");
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          filenames.push_back (std::string("solution-") +
                               Utilities::int_to_string (cycle, 5) +
                               "." +
                               Utilities::int_to_string(i, 4) +
                               ".vtu");
        const std::string
        pvtu_master_filename = ("solution-" +
                                Utilities::int_to_string (cycle, 5) +
                                ".pvtu");
        std::ofstream pvtu_master (pvtu_master_filename.c_str());
        data_out.write_pvtu_record (pvtu_master, filenames);
      }

    // level output
    if (false) for (unsigned int lvl=0; lvl<triangulation.n_global_levels(); ++lvl)
        {
          pcout << "writing lvl " << lvl << std::endl;
          const std::string filename = ("solution-" +
                                        Utilities::int_to_string (cycle, 5) +
                                        "." +
                                        Utilities::int_to_string
                                        (triangulation.locally_owned_subdomain(), 4) +
                                        "L" +
                                        Utilities::int_to_string
                                        (lvl, 2) +
                                        ".vtu");

          LvlDataOut<dim> data_out(triangulation.locally_owned_subdomain(), lvl);
          data_out.attach_dof_handler (dof_handler);
          Vector<float> subdomain (triangulation.n_active_cells());
          typename Triangulation<dim>::cell_iterator
          cell = triangulation.begin(lvl);
          for (unsigned int i=0; cell != triangulation.end(lvl); ++cell, ++i)
            subdomain = cell->level_subdomain_id();
          data_out.add_data_vector (subdomain, "lvlsubdomain");

          data_out.build_patches (0);
          std::ofstream output (filename.c_str());
          data_out.write_vtu (output);
        }


  }


  template <int dim>
  void
  InteriorPenaltyProblem<dim>::run(unsigned int n_steps)
  {
    pcout << "Element: " << fe.get_name() << std::endl;
    for (unsigned int s=0; s<n_steps; ++s)
      {
        pcout << "Step " << s << std::endl;

        if (s>0)
          {
            if (false)
              triangulation.refine_global(1);
            else
              {
                LA::MPI::Vector temp_solution;
                temp_solution.reinit(locally_relevant_set, MPI_COMM_WORLD);
                temp_solution = solution;
                Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
                KellyErrorEstimator<dim>::estimate (dof_handler,
                                                    QGauss<dim-1>(fe.degree+1),
                                                    std::map<types::boundary_id,const Function<dim> *>(),
                                                    temp_solution,
                                                    estimated_error_per_cell);


                parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
                    estimated_error_per_cell,
                    0.5, 0.0);
                triangulation.execute_coarsening_and_refinement ();
              }
          }

        pcout << "Triangulation "
              << triangulation.n_global_active_cells() << " cells, "
              << triangulation.n_global_levels() << " levels" << std::endl;

        setup_system();
        if (settings.use_amg)
          {
            pcout << "DoFHandler " << dof_handler.n_dofs() << std::endl;
          }
        else
          {
            pcout << "DoFHandler " << dof_handler.n_dofs() << " dofs, level dofs";
            for (unsigned int l=0; l<triangulation.n_global_levels(); ++l)
              pcout << ' ' << dof_handler.n_dofs(l);
            pcout << std::endl;
          }

        pcout << "Assemble matrix" << std::endl;
        assemble_matrix();
        if (!settings.use_amg)
          {
            pcout << "Assemble multilevel matrix" << std::endl;
            assemble_mg_matrix();
          }

        pcout << "Assemble right hand side" << std::endl;
        assemble_right_hand_side();
        pcout << "Solve" << std::endl;
        solve();
        output_results(s);

        pcout << std::endl;
      }
  }
}



int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef DEAL_II_WITH_PETSC
  PetscPopSignalHandler();
#endif

  using namespace dealii;
  using namespace Step39;

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog.depth_console (3);
  else
    deallog.depth_console (0);

  try
    {
      InteriorPenaltyProblem<3> test((argc>1) ? (argv[1]) : "");
      test.run(20);
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
