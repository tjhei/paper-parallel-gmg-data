#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
//#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/output.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <memory>

using namespace dealii;
using namespace LocalIntegrators;


struct TimingStat
{
  std::vector<double> times;
  double avg;
  double min;

  void clear()
  {
    times.clear();
    avg=min=-1.0;
  }

  template <class stream_T>
  void print(stream_T &out)
  {
    std::sort(times.begin(), times.end());
    min = times.front();
    out << "min= " << times.front();
    avg = 0.;
    for (auto x: times)
      {
        avg += x;
      }
    avg /= times.size();
    out << " avg= " << avg;
    out << " all:";
    for (auto x: times)
      {
        out << " " << x;
      }
    out << std::endl;
  }

};

/**
 * Matrix-free operators must use deal.II defined vectors, rest of the code is based on Trilinos vectors.
 */
namespace ChangeVectorTypes
{
  void import(TrilinosWrappers::MPI::Vector &out,
              const dealii::LinearAlgebra::ReadWriteVector<double> &rwv,
              const VectorOperation::values                 operation)
  {
    Assert(out.size() == rwv.size(),
           ExcMessage("Both vectors need to have the same size for import() to work!"));

    Assert(out.locally_owned_elements() == rwv.get_stored_elements(),
           ExcNotImplemented());

    if (operation == VectorOperation::insert)
      {
        for (const auto idx : out.locally_owned_elements())
          out[idx] = rwv[idx];
      }
    else if (operation == VectorOperation::add)
      {
        for (const auto idx : out.locally_owned_elements())
          out[idx] += rwv[idx];
      }
    else
      AssertThrow(false, ExcNotImplemented());

    out.compress(operation);
  }


  void copy(TrilinosWrappers::MPI::Vector &out,
            const dealii::LinearAlgebra::distributed::Vector<double> &in)
  {
    dealii::LinearAlgebra::ReadWriteVector<double> rwv(out.locally_owned_elements());
    rwv.import(in, VectorOperation::insert);
    //This import function doesn't exist until after dealii 9.0
    //Implemented above
    import(out, rwv,VectorOperation::insert);
  }

  void copy(dealii::LinearAlgebra::distributed::Vector<double> &out,
            const TrilinosWrappers::MPI::Vector &in)
  {
    dealii::LinearAlgebra::ReadWriteVector<double> rwv;
    rwv.reinit(in);
    out.import(rwv, VectorOperation::insert);
  }
}


template <int dim>
const Function<dim> *get_exact_solution()
{
  static ZeroFunction<3> f;
  return &f;
}

template <>
const Function<2> *get_exact_solution<2>()
{
  if ((false))
    {
      static Functions::LSingularityFunction exact_solution;
      return static_cast<const Function<2>* >(&exact_solution);
    }
  else
    {
      static ZeroFunction<2> f;
      return &f;
    }
}

template <int dim>
class RHS : public Function<dim>
{
public:
  virtual double value(const Point<dim> &/*p*/,
                       const unsigned int /*component*/ = 0) const override
  {
    return 1.0;
  }

  //  virtual void value_list(const std::vector<Point<dim>> &points,
  //                          std::vector<double> &          values,
  //                          const unsigned int component = 0) const override;
};


template <int dim>
class Coefficient : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  //  virtual void value_list(const std::vector<Point<dim>> &points,
  //                          std::vector<double> &          values,
  //                          const unsigned int component = 0) const override;

  template <typename number>
  VectorizedArray<number>
  value(const Point<dim, VectorizedArray<number>> &p,
        const unsigned int        component = 0) const;
};



template <int dim>
double Coefficient<dim>::value(const Point<dim> &p, const unsigned int) const
{
  for (int d = 0; d < dim; ++d)
    {
      if (p[d]<-0.5)
        return 100.0;
    }
  return 1.0;
  //  if (p.square() < 0.5 * 0.5)
  //    return 5;
  //  else
  //    return 1;
}


void
average(std::vector<double> &values)
{
  double sum = 0.0;
  for (unsigned int i=0; i<values.size(); ++i)
    sum+=values[i];
  sum/=values.size();

  for (unsigned int i=0; i<values.size(); ++i)
    values[i] = sum;
}


template <int dim>
template <typename number>
VectorizedArray<number>
Coefficient<dim>::value(const Point<dim,VectorizedArray<number> > &p, const unsigned int) const
{
  VectorizedArray<number> return_value = VectorizedArray<number>();
  for (unsigned int i=0; i<VectorizedArray<number>::n_array_elements; ++i)
    {
      bool found = false;
      for (int d = 0; d < dim; ++d)
        if (p[d][i]<-0.5)
          {
            return_value[i] = 100.0;
            found = true;
            break;
          }

      if (!found)
        return_value[i] = 1.0;
    }

  return return_value;
}



/**
 * Matrix-free Laplace operator
 */
template <int dim, int fe_degree, typename number>
class LaplaceOperator
  : public MatrixFreeOperators::
  Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:
  LaplaceOperator();

  void clear() override;

  void evaluate_coefficient(const Coefficient<dim> &coefficient_function);
  Table<1,VectorizedArray<number>> get_coefficient_table ();

  virtual void compute_diagonal() override;

private:
  virtual void apply_add(
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src) const override;

  void
  local_apply(const MatrixFree<dim, number>                    &data,
              LinearAlgebra::distributed::Vector<number>       &dst,
              const LinearAlgebra::distributed::Vector<number> &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const;

  void local_compute_diagonal(
    const MatrixFree<dim, number>               &data,
    LinearAlgebra::distributed::Vector<number> &dst,
    const unsigned int                          &dummy,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  Table<1, VectorizedArray<number>> coefficient;
};



template <int dim, int fe_degree, typename number>
LaplaceOperator<dim, fe_degree, number>::LaplaceOperator()
  : MatrixFreeOperators::Base<dim,
    LinearAlgebra::distributed::Vector<number>>()
{}



template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim, fe_degree, number>::clear()
{
  coefficient.reinit(TableIndices<1>(0));
  MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
      clear();
}




template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim, fe_degree, number>::evaluate_coefficient(
  const Coefficient<dim> &coefficient_function)
{
  const unsigned int n_cells = this->data->n_macro_cells();
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);

  coefficient.reinit(TableIndices<1>(n_cells));
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      phi.reinit(cell);

      VectorizedArray<number> averaged_value(0);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        averaged_value +=
          coefficient_function.value(phi.quadrature_point(q));
      averaged_value /= phi.n_q_points;

      coefficient(cell) = averaged_value;
    }
}

template <int dim, int fe_degree, typename number>
Table<1,VectorizedArray<number>>
                              LaplaceOperator<dim, fe_degree, number>::get_coefficient_table()
{
  return coefficient;
}




template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim, fe_degree, number>::local_apply(
  const MatrixFree<dim, number>                    &data,
  LinearAlgebra::distributed::Vector<number>       &dst,
  const LinearAlgebra::distributed::Vector<number> &src,
  const std::pair<unsigned int, unsigned int>      &cell_range) const
{
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      AssertDimension(coefficient.size(0), data.n_macro_cells());

      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(false, true);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_gradient(coefficient(cell) * phi.get_gradient(q), q);
      phi.integrate(false, true);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim, fe_degree, number>::apply_add(
  LinearAlgebra::distributed::Vector<number>       &dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
}



template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim, fe_degree, number>::compute_diagonal()
{
  this->inverse_diagonal_entries.reset(
    new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
  LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
    this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inverse_diagonal);
  unsigned int dummy = 0;
  this->data->cell_loop(&LaplaceOperator::local_compute_diagonal,
                        this,
                        inverse_diagonal,
                        dummy);

  this->set_constrained_entries_to_one(inverse_diagonal);

  for (unsigned int i = 0; i < inverse_diagonal.local_size(); ++i)
    {
      Assert(inverse_diagonal.local_element(i) > 0.,
             ExcMessage("No diagonal entry in a positive definite operator "
                        "should be zero"));
      inverse_diagonal.local_element(i) =
        1. / inverse_diagonal.local_element(i);
    }
}



template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim, fe_degree, number>::local_compute_diagonal(
  const MatrixFree<dim, number>              &data,
  LinearAlgebra::distributed::Vector<number> &dst,
  const unsigned int &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

  AlignedVector<VectorizedArray<number>> diagonal(phi.dofs_per_cell);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      AssertDimension(coefficient.size(0), data.n_macro_cells());

      phi.reinit(cell);
      for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
            phi.submit_dof_value(VectorizedArray<number>(), j);
          phi.submit_dof_value(make_vectorized_array<number>(1.), i);

          phi.evaluate(false, true);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_gradient(coefficient(cell) * phi.get_gradient(q),
                                q);
          phi.integrate(false, true);
          diagonal[i] = phi.get_dof_value(i);
        }
      for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        phi.submit_dof_value(diagonal[i], i);
      phi.distribute_local_to_global(dst);
    }
}


template <int dim>
class MatrixIntegrator : public MeshWorker::LocalIntegrator<dim>
{
public:
  MatrixIntegrator(const FiniteElement<dim,dim> &element, double rhs_value);
  void cell(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
  void boundary(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
  void face(MeshWorker::DoFInfo<dim> &dinfo1, MeshWorker::DoFInfo<dim> &dinfo2,
            typename MeshWorker::IntegrationInfo<dim> &info1,
            typename MeshWorker::IntegrationInfo<dim> &info2) const;

private:
  std::vector<int> base_element;
  std::vector<bool> needs_fluxes;
  double rhs_value;
};


template <int dim>
MatrixIntegrator<dim>::MatrixIntegrator(const FiniteElement<dim,dim> &element, double rhs_value)
  :
  MeshWorker::LocalIntegrator<dim>(true, true, true),
  base_element(element.n_blocks()),
  needs_fluxes(element.n_blocks()),
  rhs_value (rhs_value)
{
  for (unsigned int i=0; i<element.n_blocks(); ++i)
    {
      base_element[i] = element.block_to_base_index(i).first;
      needs_fluxes[i] = ! element.base_element(base_element[i]).conforms(FiniteElementData<dim>::H1);
    }
}


template <int dim>
void MatrixIntegrator<dim>::cell(
  MeshWorker::DoFInfo<dim> &dinfo,
  typename MeshWorker::IntegrationInfo<dim> &info) const
{
  AssertDimension(dinfo.n_matrices(), needs_fluxes.size()*needs_fluxes.size());

  for (unsigned int i=0; i<base_element.size(); ++i)
    {
      const unsigned int diagonal = i*(base_element.size()+1);
      LocalIntegrators::Laplace::cell_matrix(dinfo.matrix(diagonal,false).matrix,
                                             info.fe_values(base_element[i]));
    }

  if (dinfo.n_vectors() > 0)
    {
      // Note: this only works for constant right hand side
      static std::vector<std::vector<std::vector<double> > > rhs(base_element.size());

      for (unsigned int i=0; i<base_element.size(); ++i)
        {
          if (rhs[i].size()==0)
            {
              rhs[i].resize(info.fe_values(base_element[i]).get_fe().n_components());
              for (unsigned int j=0; j<rhs[i].size(); ++j)
                {
                  rhs[i][j].resize(info.fe_values(0).n_quadrature_points);
                  for (unsigned int k=0; k<rhs[i][j].size(); ++k)
                    rhs[i][j][k] = rhs_value;
                }
            }
          L2::L2(dinfo.vector(0).block(i), info.fe_values(base_element[i]), rhs[i]);
        }
    }
}


template <int dim>
void MatrixIntegrator<dim>::boundary(
  MeshWorker::DoFInfo<dim> &dinfo,
  typename MeshWorker::IntegrationInfo<dim> &info) const
{
  for (unsigned int i=0; i<base_element.size(); ++i)
    {
      const FEValuesBase<dim,dim> &fe = info.fe_values(base_element[i]);
      const unsigned int deg = fe.get_fe().tensor_degree();
      const unsigned int diagonal = i*(base_element.size()+1);
      if (!fe.get_fe().conforms(FiniteElementData<dim>::H1))
        LocalIntegrators::Laplace::nitsche_matrix(
          dinfo.matrix(diagonal,false).matrix, fe,
          LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg));
    }
}


template <int dim>
void MatrixIntegrator<dim>::face(
  MeshWorker::DoFInfo<dim> &dinfo1,
  MeshWorker::DoFInfo<dim> &dinfo2,
  typename MeshWorker::IntegrationInfo<dim> &info1,
  typename MeshWorker::IntegrationInfo<dim> &info2) const
{
  for (unsigned int i=0; i<base_element.size(); ++i)
    {
      const FEValuesBase<dim,dim> &fe1 = info1.fe_values(base_element[i]);
      const FEValuesBase<dim,dim> &fe2 = info2.fe_values(base_element[i]);
      const unsigned int deg = info1.fe_values(base_element[i]).get_fe().tensor_degree();
      const unsigned int diagonal = i*(base_element.size()+1);

      // The following commented lines are needed to make the broken
      // multigrid work with local refinement. As soon as hanging
      // nodes are eliminated properly, the jump terms of continuous
      // elements or continuous components add up to zero and have no
      // effect.

      // if (!fe1.get_fe().conforms(FiniteElementData<dim>::H1))
      //  if (fe1.get_fe().conforms(FiniteElementData<dim>::Hdiv))
      // LocalIntegrators::Laplace::ip_tangential_matrix(
      // dinfo1.matrix(diagonal,false).matrix, dinfo1.matrix(diagonal,true).matrix,
      // dinfo2.matrix(diagonal,true).matrix, dinfo2.matrix(diagonal,false).matrix,
      // fe1, fe2,
      // LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
      //  else
      LocalIntegrators::Laplace::ip_matrix(
        dinfo1.matrix(diagonal,false).matrix, dinfo1.matrix(diagonal,true).matrix,
        dinfo2.matrix(diagonal,true).matrix, dinfo2.matrix(diagonal,false).matrix,
        fe1, fe2,
        LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
    }
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

  RHS<dim> rhs;
  const double rhs_value = rhs.value(dinfo.cell->center());

  Coefficient<dim> coefficient;
  const double nu = coefficient.value(dinfo.cell->center());

  for (unsigned k=0; k<fe.n_quadrature_points; ++k)
    {
      const double t = dinfo.cell->diameter() * (rhs_value+nu*trace(DDuh[k]));
      dinfo.value(0) +=  t*t * fe.JxW(k);
    }
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}


template <int dim>
void Estimator<dim>::boundary(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const
{
  /*const FEValuesBase<dim> &fe = info.fe_values();

  std::vector<double> boundary_values(fe.n_quadrature_points);
  get_exact_solution<dim>()->value_list(fe.get_quadrature_points(), boundary_values);

  const std::vector<double> &uh = info.values[0][0];

  const unsigned int deg = fe.get_fe().tensor_degree();
  const double penalty = 2. * deg * (deg+1) * dinfo.face->measure() / dinfo.cell->measure();

  for (unsigned k=0; k<fe.n_quadrature_points; ++k)
    dinfo.value(0) += penalty * (boundary_values[k] - uh[k]) * (boundary_values[k] - uh[k])
                      * fe.JxW(k);
  dinfo.value(0) = std::sqrt(dinfo.value(0));
  if (!dinfo.cell->is_locally_owned())*/
  dinfo.value(0) = 0.0;
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

  Coefficient<dim> coefficient;
  const double nu1 = coefficient.value(dinfo1.cell->center());
  const double nu2 = coefficient.value(dinfo2.cell->center());

  for (unsigned k=0; k<fe.n_quadrature_points; ++k)
    {
      double diff1 = (uh1[k] - uh2[k])*(nu1+nu2)/2.0;
      double diff2 = nu1*fe.normal_vector(k) * Duh1[k] - nu2*fe.normal_vector(k) * Duh2[k];
      dinfo1.value(0) += (penalty * diff1*diff1 + h * diff2*diff2)
                         * fe.JxW(k);
    }
  dinfo1.value(0) = std::sqrt(dinfo1.value(0));
  dinfo2.value(0) = dinfo1.value(0);
  // do not fill values if cells are ghost cells because we don't communicate
  /*if (!dinfo1.cell->is_locally_owned())
    dinfo1.value(0) = 0.0;
  if (!dinfo2.cell->is_locally_owned())
    dinfo2.value(0) = 0.0;*/
}

struct Settings
{
  bool try_parse(const std::string &prm_filename);

  enum AssembleEnum
  {
    matrix_free,
    matrix_based,
    amg,
    mesh_worker
  } assembler;

  int dimension;
  std::string problem;
  std::string refinement_type;
  std::string fe;
  std::string smoother;
  double smoother_dampen;
  unsigned int smoother_steps;
  unsigned int n_steps;

  bool output;

  std::string assembler_text;
};

template <int dim>
class LaplaceProblem
{
#ifdef DEAL_II_WITH_TRILINOS
  typedef parallel::distributed::Triangulation<dim> tria_t;
  typedef TrilinosWrappers::SparseMatrix MatrixType;
  typedef TrilinosWrappers::MPI::Vector VectorType;
  typedef TrilinosWrappers::PreconditionJacobi JacobiSmoother;
  typedef TrilinosWrappers::PreconditionSSOR SSORSmoother;
  typedef TrilinosWrappers::PreconditionBlockJacobi BlockJacobiSmoother;
  typedef TrilinosWrappers::PreconditionBlockSSOR BlockSSORSmoother;

  typedef LaplaceOperator<dim,2,float> MatrixFreeLevelMatrix;
  typedef LaplaceOperator<dim,2,double> MatrixFreeSytemMatrix;
  typedef LinearAlgebra::distributed::Vector<float> MatrixFreeLevelVector;
  typedef LinearAlgebra::distributed::Vector<double> MatrixFreeSystemVector;


#else
  typedef Triangulation<dim> tria_t;
  typedef SparseMatrix<double> MatrixType;
  typedef Vector<double> VectorType;
  typedef PreconditionJacobi<MatrixType> JacobiSmoother;
  typedef PreconditionSSOR<MatrixType> SSORSmoother;
#endif

public:
  LaplaceProblem(const Settings &settings);
  void run();

private:
  void setup_system ();
  void assemble_system ();
  void assemble_system_mw ();
  void assemble_multigrid ();
  void assemble_multigrid_mw ();
  void assemble_rhs_for_matrix_free ();
  void error ();
  void solve ();
  void estimate ();
  void refine_grid ();
  void output_results (const unsigned int cycle);

  Settings settings;

  std::vector<std::pair<std::string, double>> stats;

  MPI_Comm           mpi_communicator;
  ConditionalOStream                        pcout;

  tria_t triangulation;
  const MappingQ1<dim> mapping;
  std::unique_ptr<const FiniteElement<dim> > fe;
  DoFHandler<dim> dof_handler;


  SparsityPattern seq_sparsity;
  MatrixType system_matrix;

  MatrixFreeSytemMatrix mf_system_matrix;

  IndexSet locally_relevant_set;
  AffineConstraints<double> constraints;

  VectorType solution;
  VectorType right_hand_side;
  BlockVector<double> estimate_vector;

  const unsigned int degree;

  MGLevelObject<SparsityPattern> seq_mg_sparsity;
  MGLevelObject<SparsityPattern> seq_mg_sparsity_interface;

  MGLevelObject<MatrixType> mg_matrix;
  MGLevelObject<MatrixType> mg_matrix_dg_down;
  MGLevelObject<MatrixType> mg_matrix_dg_up;
  MGLevelObject<MatrixType> mg_interface_in;
  MGLevelObject<MatrixType> mg_interface_out;

  MGLevelObject<MatrixFreeLevelMatrix> mf_mg_matrix;

  MGConstrainedDoFs mg_constrained_dofs;

  TimerOutput computing_timer;
};


template <int dim>
LaplaceProblem<dim>::LaplaceProblem(const Settings &settings)
  :
  settings (settings),
  mpi_communicator(MPI_COMM_WORLD),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(mpi_communicator)
          == 0)),
#ifdef DEAL_II_WITH_TRILINOS
  triangulation (mpi_communicator,Triangulation<dim>::
                 limit_level_difference_at_vertices,
                 (settings.assembler==Settings::amg) ?
                   tria_t::default_setting
                 :
                 tria_t::construct_multigrid_hierarchy),
#else
  triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
#endif
  mapping(),
  fe(FETools::get_fe_by_name<dim,dim>(settings.fe)),
  dof_handler(triangulation),
  estimate_vector(1),
  degree(fe->tensor_degree()),
  computing_timer (pcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times)
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator)!=0)
    deallog.depth_console(0);

  if (settings.problem == "hyper_L")
    {
      GridGenerator::hyper_L(triangulation, -1, 1, /*colorize*/ false);
      triangulation.refine_global(1);
    }
  else if (settings.problem == "hyper_cube")
    {
      GridGenerator::hyper_cube(triangulation, -1, 1);
      triangulation.refine_global();
    }
  else if (settings.problem == "minimal")
    {
      std::vector<unsigned int> sub = {2,1};
      Point<dim> p1;
      Point<dim> p2;
      p2(0) = 2.;
      p2(1) = 1.;
      GridGenerator::subdivided_hyper_rectangle(triangulation, sub,p1,p2);
      triangulation.begin()->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
    }
  else
    throw ExcNotImplemented();
}


bool
Settings::try_parse(const std::string &prm_filename)
{
  ParameterHandler prm;
  prm.declare_entry("dim", "2",
                    Patterns::Integer(),
                    "");
  prm.declare_entry("problem", "hyper_L",
                    Patterns::Selection("hyper_L"),
                    "Select problem to solve");
  prm.declare_entry("refinement type", "kelly",
                    Patterns::Selection("global|circle|first quadrant|kelly|estimator"),
                    "Select how to refine. Options: global|circle|first quadrant|kelly|estimator");
  prm.declare_entry("FE", "FE_Q<2>(2)",
                    Patterns::Anything(),
                    "Select your FiniteElement");
  prm.declare_entry("n_steps", "20",
                    Patterns::Integer(0),
                    "Number of adaptive refinement steps.");
  prm.declare_entry("smoother", "jacobi",
                    Patterns::Selection("jacobi|ssor|block jacobi|block ssor"),
                    "");
  prm.declare_entry("smoother dampen", "1.0",
                    Patterns::Double(0.0),
                    "");
  prm.declare_entry("smoother steps", "2",
                    Patterns::Integer(1),
                    "");
  prm.declare_entry("assembler", "matrix based",
                    Patterns::Selection("matrix free|matrix based|AMG|mesh worker"),
                    "");

  prm.declare_entry("output", "false",
                    Patterns::Bool(),
                    "");

  try
    {
      prm.parse_input(prm_filename);
    }
  catch (...)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
        prm.print_parameters(std::cout, ParameterHandler::Text);
      return false;
    }
  this->problem = prm.get("problem");
  this->dimension = prm.get_integer("dim");
  this->fe = prm.get("FE");
  this->refinement_type = prm.get("refinement type");
  this->n_steps = prm.get_integer("n_steps");
  this->smoother = prm.get("smoother");
  this->smoother_dampen = prm.get_double("smoother dampen");
  this->smoother_steps = prm.get_integer("smoother steps");

  if (prm.get("assembler")=="matrix free")
    this->assembler = matrix_free;
  else if (prm.get("assembler")=="matrix based")
    this->assembler = matrix_based;
  else if (prm.get("assembler")=="AMG")
    this->assembler = amg;
  else if (prm.get("assembler")=="mesh worker")
    this->assembler = mesh_worker;
  else AssertThrow(false, ExcNotImplemented());
  this->assembler_text = prm.get("assembler");

  this->output = prm.get_bool("output");

  return true;
}


template <int dim>
void
LaplaceProblem<dim>::setup_system()
{
  TimerOutput::Scope timing (computing_timer, "Setup");
  {
    TimerOutput::Scope timing (computing_timer, "Setup: distribute_dofs");
    dof_handler.distribute_dofs(*fe);
  }

  if (settings.assembler != Settings::amg)
  {
    TimerOutput::Scope timing (computing_timer, "Setup: distribute_mg_dofs");
    dof_handler.distribute_mg_dofs();
  }

  deallog << "Number of degrees of freedom: " << dof_handler.n_dofs();

  if (settings.assembler != Settings::amg)
    {
      for (unsigned int l=0; l<triangulation.n_global_levels(); ++l)
        deallog << "   " << 'L' << l << ": "
                << dof_handler.n_dofs(l);
    }
  deallog  << std::endl;

  if (settings.assembler != Settings::amg)
    deallog << "workload imbalance: "
	    << MGTools::workload_imbalance(triangulation) << std::endl;

  computing_timer.enter_section("Setup: Sparsity, vectors, and MF");

  DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_set);

#ifdef DEAL_II_WITH_TRILINOS
  solution.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
  right_hand_side.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
#else
  solution.reinit(dof_handler.n_dofs());
  right_hand_side.reinit(dof_handler.n_dofs());
#endif
  constraints.reinit (locally_relevant_set);
  DoFTools::make_hanging_node_constraints (dof_handler, constraints);

  if (settings.problem == "hyper_L")
    VectorTools::interpolate_boundary_values (mapping, dof_handler, 0,
                                              *(get_exact_solution<dim>()),
                                              constraints);
  else
    DoFTools::make_zero_boundary_constraints(dof_handler, constraints);



  constraints.close ();

  if (settings.assembler == Settings::matrix_based
      ||
      settings.assembler == Settings::amg)
    {
#ifdef DEAL_II_WITH_TRILINOS
      TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(),
                                            dof_handler.locally_owned_dofs(),
                                            locally_relevant_set,
                                            MPI_COMM_WORLD);
#else
      DynamicSparsityPattern dsp(dof_handler.n_dofs(),
                                 dof_handler.n_dofs(),
                                 locally_relevant_set);
#endif

      //DoFTools::make_flux_sparsity_pattern (dof_handler, dsp, constraints);
      DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints);

#ifdef DEAL_II_WITH_TRILINOS
      dsp.compress();
      system_matrix.reinit (dsp);
#else
      seq_sparsity.copy_from(dsp);
      system_matrix.reinit (seq_sparsity);
#endif
    }
  else if (settings.assembler == Settings::matrix_free)
    {
      Assert (fe->degree == 2, ExcMessage("Needs simple implementation"));

      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points);
      std::shared_ptr<MatrixFree<dim, double>> mf_storage(
                                              new MatrixFree<dim, double>());
      mf_storage->reinit(dof_handler,
                         constraints,
                         QGauss<1>(fe->degree + 1),
                         additional_data);
      mf_system_matrix.initialize(mf_storage);
    }

  computing_timer.exit_section("Setup: Sparsity, vectors, and MF");


  if (settings.assembler == Settings::amg)
    return;


  computing_timer.enter_section("Setup: GMG sparsity and MF");

  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler);
  std::set<types::boundary_id> bset;
  bset.insert(0);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, bset);

  if (settings.assembler == Settings::matrix_free)
    {
      const unsigned int nlevels = triangulation.n_global_levels();
      mf_mg_matrix.resize(0, nlevels - 1);

      for (unsigned int level = 0; level < nlevels; ++level)
        {
          IndexSet relevant_dofs;
          DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                        level,
                                                        relevant_dofs);
          AffineConstraints<double> level_constraints;
          level_constraints.reinit(relevant_dofs);
          level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
          level_constraints.close();

          typename MatrixFree<dim, float>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
            MatrixFree<dim, float>::AdditionalData::none;
          additional_data.mapping_update_flags =
            (update_gradients | update_JxW_values | update_quadrature_points);
          additional_data.mg_level = level;
          std::shared_ptr<MatrixFree<dim, float>> mf_storage_level(
                                                 new MatrixFree<dim, float>());
          mf_storage_level->reinit(dof_handler,
                                   level_constraints,
                                   QGauss<1>(fe->degree + 1),
                                   additional_data);

          mf_mg_matrix[level].initialize(mf_storage_level,
                                         mg_constrained_dofs,
                                         level);

          mf_mg_matrix[level].evaluate_coefficient(Coefficient<dim>());
          mf_mg_matrix[level].compute_diagonal();
        }

      computing_timer.exit_section("Setup: GMG sparsity and MF");

      return;
    }



  const unsigned int n_levels = triangulation.n_global_levels();
  mg_matrix.resize(0, n_levels-1);
  mg_matrix.clear_elements();
  //mg_matrix_dg_up.resize(0, n_levels-1);
  //mg_matrix_dg_up.clear_elements();
  //mg_matrix_dg_down.resize(0, n_levels-1);
  //mg_matrix_dg_down.clear_elements();
  mg_interface_in.resize(0, n_levels-1);
  mg_interface_in.clear_elements ();
  //mg_interface_out.resize(0, n_levels-1);
  //mg_interface_out.clear_elements ();

  //seq_mg_sparsity.resize(0, n_levels-1);
  //seq_mg_sparsity_interface.resize(0, n_levels-1);

  for (unsigned int level = 0; level < n_levels; ++level)
    {
      IndexSet dofset;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                    level,
                                                    dofset);
      {
#ifdef DEAL_II_WITH_TRILINOS
        TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_mg_dofs(level),
                                              dof_handler.locally_owned_mg_dofs(level),
                                              dofset,
                                              mpi_communicator);
#else
        DynamicSparsityPattern dsp(dof_handler.n_dofs(level),
                                   dof_handler.n_dofs(level),
                                   dofset);
#endif
        MGTools::make_sparsity_pattern(dof_handler, dsp, level);

#ifdef DEAL_II_WITH_TRILINOS
        dsp.compress();
        mg_matrix[level].reinit(dsp);
#else
        mg_matrix[level].reinit(dof_handler.locally_owned_mg_dofs(level),
                                dof_handler.locally_owned_mg_dofs(level),
                                dsp,
                                mpi_communicator,
                                true);
#endif
      }

      {
#ifdef DEAL_II_WITH_TRILINOS
        TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_mg_dofs(level),
                                              dof_handler.locally_owned_mg_dofs(level),
                                              dofset,
                                              mpi_communicator);
#else
        DynamicSparsityPattern dsp(dof_handler.n_dofs(level),
                                   dof_handler.n_dofs(level),
                                   dofset);
#endif
        MGTools::make_interface_sparsity_pattern(dof_handler,
                                                 mg_constrained_dofs,
                                                 dsp,
                                                 level);

#ifdef DEAL_II_WITH_TRILINOS
        dsp.compress();
        mg_interface_in[level].reinit(dsp);
#else
        mg_interface_in[level].reinit(
          dof_handler.locally_owned_mg_dofs(level),
          dof_handler.locally_owned_mg_dofs(level),
          dsp,
          mpi_communicator,
          true);
#endif
        /*mg_interface_out[level].reinit(
            dof_handler.locally_owned_mg_dofs(level),
            dof_handler.locally_owned_mg_dofs(level),
            dsp,
            mpi_communicator,
            true);*/
      }
    }

  computing_timer.exit_section("Setup: GMG sparsity and MF");



  if (false) // for comparison with example_dg
    {
      TimerOutput::Scope timing2 (computing_timer, "Setup MG");

      for (unsigned int level=0; level<n_levels; ++level)
        {
          DynamicSparsityPattern dsp(dof_handler.n_dofs(level));
          MGTools::make_flux_sparsity_pattern(dof_handler, dsp, level);
#ifdef DEAL_II_WITH_TRILINOS
          //          mg_matrix[level].reinit(dof_handler.locally_owned_mg_dofs(level),
          //                                  dof_handler.locally_owned_mg_dofs(level),
          //                                  dsp,
          //                                  MPI_COMM_WORLD, true);

          //          mg_interface_in[level].reinit(dof_handler.locally_owned_mg_dofs(level),
          //                                        dof_handler.locally_owned_mg_dofs(level),
          //                                        dsp,
          //                                        MPI_COMM_WORLD, true);
          //          mg_interface_out[level].reinit(dof_handler.locally_owned_mg_dofs(level),
          //                                         dof_handler.locally_owned_mg_dofs(level),
          //                                         dsp,
          //                                         MPI_COMM_WORLD, true);
#else
          seq_mg_sparsity[level].copy_from(dsp);
          mg_matrix[level].reinit(seq_mg_sparsity[level]);
          mg_interface_in[level].reinit(seq_mg_sparsity[level]);
          mg_interface_out[level].reinit(seq_mg_sparsity[level]);
#endif
          if (level>0)
            {
              DynamicSparsityPattern dsp;
              dsp.reinit(dof_handler.n_dofs(level-1), dof_handler.n_dofs(level));
              MGTools::make_flux_sparsity_pattern_edge(dof_handler, dsp, level);
#ifdef DEAL_II_WITH_TRILINOS
              mg_matrix_dg_up[level].reinit(dof_handler.locally_owned_mg_dofs(level-1),
                                            dof_handler.locally_owned_mg_dofs(level),
                                            dsp,
                                            MPI_COMM_WORLD, true);
              mg_matrix_dg_down[level].reinit(dof_handler.locally_owned_mg_dofs(level-1),
                                              dof_handler.locally_owned_mg_dofs(level),
                                              dsp,
                                              MPI_COMM_WORLD, true);
#else
              seq_mg_sparsity_interface[level].copy_from(dsp);
              mg_matrix_dg_up[level].reinit(seq_mg_sparsity_interface[level]);
              mg_matrix_dg_down[level].reinit(seq_mg_sparsity_interface[level]);
#endif
            }
        }
    }
}


template <int dim>
void
LaplaceProblem<dim>::assemble_system()
{
  TimerOutput::Scope timing (computing_timer, "Assemble");

  const QGauss<dim> quadrature_formula(degree + 1);

  FEValues<dim> fe_values(*fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const Coefficient<dim> coefficient;
  std::vector<double>    coefficient_values(n_q_points);
  RHS<dim> rhs;
  std::vector<double>    rhs_values(n_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);

        coefficient.value_list(fe_values.get_quadrature_points(),
                               coefficient_values);
        rhs.value_list(fe_values.get_quadrature_points(),
                       rhs_values);

        average(coefficient_values);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  (coefficient_values[q_point] *
                   fe_values.shape_grad(i, q_point) *
                   fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

              cell_rhs(i) += (fe_values.shape_value(i, q_point) * rhs_values[q_point] *
                              fe_values.JxW(q_point));
            }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               right_hand_side);
      }

  system_matrix.compress(VectorOperation::add);
  right_hand_side.compress(VectorOperation::add);
}


template <int dim>
void
LaplaceProblem<dim>::assemble_system_mw()
{
  TimerOutput::Scope timing (computing_timer, "Assemble");
  MeshWorker::IntegrationInfoBox<dim> info_box;
  UpdateFlags update_flags = update_values | update_gradients;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*fe, mapping, &dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(dof_handler.block_info());

  MeshWorker::Assembler::SystemSimple<MatrixType, VectorType> assembler;
  assembler.initialize(constraints);
  assembler.initialize(system_matrix, right_hand_side);

  double rhs_value = 0.;//((settings.problem=="hyper_L") ? 0.0 : 1.0);
  MatrixIntegrator<dim> integrator(dof_handler.get_fe(), rhs_value);
  MeshWorker::integration_loop<dim, dim> (
    dof_handler.begin_active(), dof_handler.end(),
    dof_info, info_box, integrator, assembler);

  system_matrix.compress(VectorOperation::add);
  right_hand_side.compress(VectorOperation::add);
  // TODO: not efficient
  for (unsigned int i=0; i<dof_handler.locally_owned_dofs().n_elements(); ++i)
    {
      types::global_dof_index idx = dof_handler.locally_owned_dofs().nth_index_in_set(i);
      if (constraints.is_constrained(idx))
        system_matrix.set(idx,idx,1.);
    }
  system_matrix.compress(VectorOperation::insert);
}


template <int dim>
void
LaplaceProblem<dim>::assemble_multigrid()
{
  TimerOutput::Scope timing (computing_timer, "Assemble MG");

  QGauss<dim> quadrature_formula(1 + degree);

  FEValues<dim> fe_values(*fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const Coefficient<dim> coefficient;
  std::vector<double>    coefficient_values(n_q_points);

  std::vector<AffineConstraints<double>> boundary_constraints(
                                        triangulation.n_global_levels());
  for (unsigned int level = 0; level < triangulation.n_global_levels();
       ++level)
    {
      IndexSet dofset;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                    level,
                                                    dofset);
      boundary_constraints[level].reinit(dofset);
      boundary_constraints[level].add_lines(
        mg_constrained_dofs.get_refinement_edge_indices(level));
      boundary_constraints[level].add_lines(
        mg_constrained_dofs.get_boundary_indices(level));

      boundary_constraints[level].close();
    }

  for (const auto &cell : dof_handler.cell_iterators())
    if (cell->level_subdomain_id() == triangulation.locally_owned_subdomain())
      {
        cell_matrix = 0;
        fe_values.reinit(cell);

        coefficient.value_list(fe_values.get_quadrature_points(),
                               coefficient_values);

        average(coefficient_values);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) +=
                (coefficient_values[q_point] *
                 fe_values.shape_grad(i, q_point) *
                 fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

        cell->get_mg_dof_indices(local_dof_indices);

        boundary_constraints[cell->level()].distribute_local_to_global(
          cell_matrix, local_dof_indices, mg_matrix[cell->level()]);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            if (mg_constrained_dofs.is_interface_matrix_entry(
                  cell->level(), local_dof_indices[i], local_dof_indices[j]))
              mg_interface_in[cell->level()].add(local_dof_indices[i],
                                                 local_dof_indices[j],
                                                 cell_matrix(i, j));
      }

  for (unsigned int i = 0; i < triangulation.n_global_levels(); ++i)
    {
      mg_matrix[i].compress(VectorOperation::add);
      mg_interface_in[i].compress(VectorOperation::add);
    }

}

template <int dim>
void
LaplaceProblem<dim>::assemble_multigrid_mw()
{
  TimerOutput::Scope timing (computing_timer, "Assemble MG");
  MeshWorker::IntegrationInfoBox<dim> info_box;
  UpdateFlags update_flags = update_values | update_gradients;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*fe, mapping, &dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(dof_handler.block_info());

  MeshWorker::Assembler::MGMatrixSimple<MatrixType> assembler;
  assembler.initialize(mg_constrained_dofs);
  assembler.initialize(mg_matrix);
  assembler.initialize_interfaces(mg_interface_in, mg_interface_in);
  assembler.initialize_fluxes(mg_matrix_dg_up, mg_matrix_dg_down);

  double rhs_value = ((settings.problem=="hyper_L") ? 0.0 : 1.0);
  MatrixIntegrator<dim> integrator(dof_handler.get_fe(), rhs_value);
  MeshWorker::integration_loop<dim, dim> (
    dof_handler.begin_mg(), dof_handler.end_mg(),
    dof_info, info_box, integrator, assembler);

  for (unsigned int level=mg_matrix.min_level(); level <= mg_matrix.max_level(); ++level)
    {
      mg_matrix[level].compress(VectorOperation::add);

      // TODO: not efficient
      const IndexSet &bdry_set = mg_constrained_dofs.get_boundary_indices(level);
      for (IndexSet::ElementIterator it = bdry_set.begin();
           it != bdry_set.end(); ++it)
        {
          if (dof_handler.locally_owned_mg_dofs(level).is_element(*it))
            mg_matrix[level].set(*it,*it,1.);
        }

      // The folowing loop was not in the parallel version. Why did
      // ythis work? There are zeros on the diagonal. But the values
      // generated are not used in the algorithm. So, Trilinos's
      // Jacobi preconditioner did not bother?
      const IndexSet &edge_set = mg_constrained_dofs.get_refinement_edge_indices(level);
      for (IndexSet::ElementIterator it = edge_set.begin();
           it != edge_set.end(); ++it)
        {
          if (dof_handler.locally_owned_mg_dofs(level).is_element(*it))
            mg_matrix[level].set(*it,*it,1.);
        }
      mg_matrix[level].compress(VectorOperation::insert);
    }
  for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
    {
      if (level > mg_matrix.min_level())
        {
          mg_matrix_dg_up[level].compress(VectorOperation::add);
          mg_matrix_dg_down[level].compress(VectorOperation::add);
        }
      mg_interface_in[level].compress(VectorOperation::add);
      //mg_interface_out[level].compress(VectorOperation::add);
    }

  // Output of matrices for minimal configuration
  if (triangulation.n_global_levels() == 2 && triangulation.n_cells(0) == 2)
    {
#ifndef DEAL_II_WITH_TRILINOS
      std::cout << "Level 0" << std::endl;
      mg_matrix[0].print_formatted(std::cout, 2, false, 5, "*");
      std::cout << "Level 1" << std::endl;
      mg_matrix[1].print_formatted(std::cout, 2, false, 5, "*");

      std::cout << "in" << std::endl;
      mg_interface_in[1].print_formatted(std::cout, 2, false, 5, "*");
      std::cout << "out" << std::endl;
      mg_interface_out[1].print_formatted(std::cout, 2, false, 5, "*");

      std::cout << "DG up" << std::endl;
      mg_matrix_dg_up[1].print_formatted(std::cout, 2, false, 5, "*");
      std::cout << "DG down" << std::endl;
      mg_matrix_dg_down[1].print_formatted(std::cout, 2, false, 5, "*");
#endif
    }
}



template <int dim>
void
LaplaceProblem<dim>::assemble_rhs_for_matrix_free()
{
  TimerOutput::Scope timing (computing_timer, "Assemble RHS");

  mf_system_matrix.evaluate_coefficient(Coefficient<dim>());

  MatrixFreeSystemVector solution_copy;
  MatrixFreeSystemVector right_hand_side_copy;
  mf_system_matrix.initialize_dof_vector(solution_copy);
  mf_system_matrix.initialize_dof_vector(right_hand_side_copy);

  solution_copy = 0.;
  constraints.distribute(solution_copy);
  solution_copy.update_ghost_values();
  right_hand_side_copy = 0;
  const Table<1, VectorizedArray<double>> coefficient
      = mf_system_matrix.get_coefficient_table();

  RHS<dim> right_hand_side_function;

  FEEvaluation<dim,2,3,1,double>
  phi (*mf_system_matrix.get_matrix_free());

  for (unsigned int cell=0; cell<mf_system_matrix.get_matrix_free()->n_macro_cells(); ++cell)
    {
      phi.reinit (cell);
      phi.read_dof_values_plain (solution_copy);
      phi.evaluate (false,true,false);

      for (unsigned int q=0; q<phi.n_q_points; ++q)
        {
          // Submit gradient
          phi.submit_gradient(-1.0*(coefficient(cell)*phi.get_gradient(q)), q);

          // Submit RHS value
          VectorizedArray<double> rhs_value = make_vectorized_array<double> (1.0);;
          for (unsigned int i=0; i<VectorizedArray<double>::n_array_elements; ++i)
            {
              Point<dim> p;
              for (unsigned int d=0; d<dim; ++d)
                p(d)=phi.quadrature_point(q)(d)[i];

              rhs_value[i] = right_hand_side_function.value(p);
            }
          phi.submit_value(rhs_value, q);
        }

      phi.integrate (true,true);
      phi.distribute_local_to_global (right_hand_side_copy);
    }

  right_hand_side_copy.compress(VectorOperation::add);
  ChangeVectorTypes::copy(right_hand_side,right_hand_side_copy);
}


template <int dim>
void
LaplaceProblem<dim>::solve()
{
  unsigned int n_timings = 5;

  TimerOutput::Scope timing (computing_timer, "Solve");

  SolverControl solver_control(1000, 1.e-10*right_hand_side.l2_norm());
  solver_control.enable_history_data();
  SolverCG<VectorType> solver(solver_control);
  SolverCG<MatrixFreeSystemVector> mf_solver(solver_control);

  solution = 0.;

  if (settings.assembler==Settings::amg)
    {
      computing_timer.enter_section("Solve: AMG Setup");

      // code to optionally compare to Trilinos ML
      typedef TrilinosWrappers::PreconditionAMG PreconditionAMG;
      //typedef TrilinosWrappers::PreconditionAMGMueLu PreconditionAMG;
      PreconditionAMG prec;

      PreconditionAMG::AdditionalData Amg_data;
      //    Amg_data.constant_modes = constant_modes;
      Amg_data.elliptic = true;
      Amg_data.smoother_type = "Jacobi";
      //Amg_data.higher_order_elements = (degree==1 ? false : true);
      Amg_data.smoother_sweeps = settings.smoother_steps;
      Amg_data.aggregation_threshold = 0.02;
      Amg_data.output_details = true;

      prec.initialize (system_matrix,
                       Amg_data);
      computing_timer.exit_section("Solve: AMG Setup");

      {
        TimingStat ts;
        Timer timer(mpi_communicator, true);

        for (unsigned int c=0; c<n_timings; ++c)
          {
            solution = 0.;
            timer.reset();
            timer.start();
            prec.vmult(solution, right_hand_side);
            timer.stop();
            ts.times.push_back(timer.wall_time());
          }

        pcout << "TS prec-vmult: ";
        ts.print(pcout);
        stats.emplace_back("prec-vmult-min", ts.min);
        stats.emplace_back("prec-vmult-avg", ts.avg);

        solution = 0.;
      }

      {

        {
          TimerOutput::Scope timing (computing_timer, "Solve: prec-vmult");
          prec.vmult(solution, right_hand_side);
        }
        solution = 0.;
      }

      {
        TimingStat ts;
        Timer timer(mpi_communicator, true);

        for (unsigned int c=0; c<n_timings; ++c)
          {
            solution = 0.;
            timer.reset();
            timer.start();
            solver.solve (system_matrix, solution, right_hand_side, prec);
            timer.stop();
            ts.times.push_back(timer.wall_time());
          }

        pcout << "TS solve: ";
        ts.print(pcout);
        stats.emplace_back("solve-min", ts.min);
        stats.emplace_back("solve-avg", ts.avg);

        solution = 0.;
      }

      {
        TimerOutput::Scope timing (computing_timer, "Solve: CG");
        solver.solve (system_matrix, solution, right_hand_side, prec);
      }
      constraints.distribute (solution);
    }
  else if (settings.assembler==Settings::matrix_free)
    {
      computing_timer.enter_section("Solve: MF Prec Setup");

      MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
      mg_transfer.build(dof_handler);

      SolverControl coarse_solver_control (1000, 1e-12, false, false);
      SolverCG<MatrixFreeLevelVector> coarse_solver(coarse_solver_control);
      PreconditionIdentity identity;
      MGCoarseGridBase<MatrixFreeLevelVector> *coarse_grid_solver = nullptr;

      MGCoarseGridIterativeSolver<MatrixFreeLevelVector, SolverCG<MatrixFreeLevelVector>,
                                  MatrixFreeLevelMatrix, PreconditionIdentity>
                                  coarse_grid_solver1(coarse_solver, mf_mg_matrix[0], identity);
      coarse_grid_solver = &coarse_grid_solver1;

      MGSmoother<MatrixFreeLevelVector> *smoother = nullptr;

      if (settings.smoother=="jacobi")
        {
          typedef PreconditionJacobi<MatrixFreeLevelMatrix> Smoother;
          auto *mg_smoother
            = new MGSmootherPrecondition<MatrixFreeLevelMatrix, Smoother, MatrixFreeLevelVector>();
          mg_smoother->initialize(mf_mg_matrix, typename Smoother::AdditionalData(settings.smoother_dampen));
          mg_smoother->set_steps(settings.smoother_steps);
          mg_smoother->set_debug(0);
          mg_smoother->set_symmetric(false); // not necessary for Jacobi
          smoother = mg_smoother;
        }
      else
        Assert(false, ExcNotImplemented());

      mg::Matrix<MatrixFreeLevelVector> mg_m(mf_mg_matrix);

      MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<MatrixFreeLevelMatrix>>
          mg_interface_matrices;
      mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
      for (unsigned int level = 0; level < triangulation.n_global_levels();
           ++level)
        mg_interface_matrices[level].initialize(mf_mg_matrix[level]);
      mg::Matrix<MatrixFreeLevelVector> mg_interface(mg_interface_matrices);

      {
        Multigrid<MatrixFreeLevelVector> mg(mg_m,
                                            *coarse_grid_solver,
                                            mg_transfer,
                                            *smoother,
                                            *smoother);
        mg.set_edge_matrices(mg_interface, mg_interface);

        if (!fe->conforms(FiniteElementData<dim>::H1))
          {
            Assert(false, ExcNotImplemented());
            //mg.set_edge_flux_matrices(mg_down, mg_up);
          }
        //mg.set_edge_matrices(mg_out, mg_in);

        PreconditionMG<dim,MatrixFreeLevelVector,MGTransferMatrixFree<dim, float>>
            preconditioner(dof_handler, mg, mg_transfer);

        MatrixFreeSystemVector solution_copy;
        MatrixFreeSystemVector right_hand_side_copy;
        mf_system_matrix.initialize_dof_vector(solution_copy);
        mf_system_matrix.initialize_dof_vector(right_hand_side_copy);

        computing_timer.exit_section("Solve: MF Prec Setup");

        ChangeVectorTypes::copy(solution_copy,solution);
        ChangeVectorTypes::copy(right_hand_side_copy,right_hand_side);

        {
          TimingStat ts;
          Timer timer(mpi_communicator, true);

          for (unsigned int c=0; c<n_timings; ++c)
            {
              solution_copy = 0.;
              timer.reset();
              timer.start();
              preconditioner.vmult(solution_copy, right_hand_side_copy);
              timer.stop();
              ts.times.push_back(timer.wall_time());
            }

          pcout << "TS prec-vmult: ";
          ts.print(pcout);
          stats.emplace_back("prec-vmult-min", ts.min);
          stats.emplace_back("prec-vmult-avg", ts.avg);

          solution_copy = 0.;
        }

        {
          {
            TimerOutput::Scope timing (computing_timer, "Solve: prec-vmult");
            preconditioner.vmult(solution_copy, right_hand_side_copy);
          }
          solution_copy = 0.;
        }

        {
          TimingStat ts;
          Timer timer(mpi_communicator, true);

          for (unsigned int c=0; c<n_timings; ++c)
            {
              solution_copy = 0.;
              timer.reset();
              timer.start();
              mf_solver.solve(mf_system_matrix, solution_copy, right_hand_side_copy, preconditioner);
              timer.stop();
              ts.times.push_back(timer.wall_time());
            }

          pcout << "TS solve: ";
          ts.print(pcout);
          stats.emplace_back("solve-min", ts.min);
          stats.emplace_back("solve-avg", ts.avg);

          solution_copy = 0.;
        }


//        std::vector<Timer> timer_presmoothing(triangulation.n_global_levels());

//        auto handler_ps = [&](const bool before, const unsigned int level)
//        {
//          if (before) timer_presmoothing[level].start();
//          else timer_presmoothing[level].stop();
//        };

//        mg.connect_pre_smoother_step(handler_ps);
        {
          TimerOutput::Scope timing (computing_timer, "Solve: CG");
          mf_solver.solve(mf_system_matrix, solution_copy, right_hand_side_copy, preconditioner);
        }
        solution_copy.update_ghost_values();
        ChangeVectorTypes::copy(solution,solution_copy);

        /*  for (unsigned int l=0;l<triangulation.n_global_levels();++l)
            std::cout << "Time pre: " << l
                << "\twall " << timer_presmoothing[l].wall_time()
                << "\tCPU  " << timer_presmoothing[l].cpu_time()
                << std::endl;*/
        constraints.distribute (solution);
      }
      delete smoother;
    }
  else if (false)
    {

      MGTransferPrebuilt<VectorType> mg_transfer(mg_constrained_dofs);
      // Now the prolongation matrix has to be built.
      mg_transfer.build_matrices(dof_handler);

      MatrixType &coarse_matrix = mg_matrix[0];

      SolverControl        coarse_solver_control(1000, 1e-10, false, false);
      SolverCG<VectorType> coarse_solver(coarse_solver_control);

      PreconditionIdentity prec;

      MGCoarseGridIterativeSolver<VectorType,
                                  SolverCG<VectorType>,
                                  MatrixType,
                                  PreconditionIdentity>
                                  coarse_grid_solver(coarse_solver, coarse_matrix, prec);
      //using Smoother = LinearAlgebra::MPI::PreconditionJacobi;
      using Smoother = JacobiSmoother;
      MGSmootherPrecondition<MatrixType, Smoother, VectorType> mg_smoother;
      mg_smoother.initialize(mg_matrix, Smoother::AdditionalData(settings.smoother_dampen));
      mg_smoother.set_steps(settings.smoother_steps);
      // mg_smoother.set_symmetric(false);

      mg::Matrix<VectorType> mg_matrix_local(mg_matrix);
      mg::Matrix<VectorType> mg_interface_up(mg_interface_in);
      mg::Matrix<VectorType> mg_interface_down(mg_interface_in);

      // Now, we are ready to set up the
      // V-cycle operator and the
      // multilevel preconditioner.
      Multigrid<VectorType> mg(
        mg_matrix_local, coarse_grid_solver, mg_transfer, mg_smoother, mg_smoother);
      // mg.set_debug(6);
      mg.set_edge_matrices(mg_interface_down, mg_interface_up);

      PreconditionMG<dim, VectorType, MGTransferPrebuilt<VectorType>>
          preconditioner(dof_handler, mg, mg_transfer);

      solver.solve(system_matrix, solution, right_hand_side, preconditioner);

      constraints.distribute(solution);

    }
  else if (true)
    {
      // matrix-based
      computing_timer.enter_section("Solve: MB Prec Setup");

      MGTransferPrebuilt<VectorType> mg_transfer(mg_constrained_dofs);
      mg_transfer.build_matrices(dof_handler);

      MatrixType &coarse_matrix = mg_matrix[0];
      SolverControl coarse_solver_control (1000, 1e-12, false, false);
      SolverCG<VectorType> coarse_solver(coarse_solver_control);
      PreconditionIdentity identity;
      MGCoarseGridBase<VectorType> *coarse_grid_solver = nullptr;

      MGCoarseGridIterativeSolver<VectorType, SolverCG<VectorType>,
                                  MatrixType, PreconditionIdentity>
                                  coarse_grid_solver1(coarse_solver, coarse_matrix, identity);
      coarse_grid_solver = &coarse_grid_solver1;

      MGSmoother<VectorType> *smoother = nullptr;

      int block_size = fe->n_dofs_per_cell();

      if (settings.smoother=="jacobi")
        {
          typedef JacobiSmoother Smoother;
          auto *mg_smoother
            = new MGSmootherPrecondition<MatrixType, Smoother, VectorType>();
          mg_smoother->initialize(mg_matrix, settings.smoother_dampen);
          mg_smoother->set_steps(settings.smoother_steps);
          mg_smoother->set_debug(0);
          mg_smoother->set_symmetric(false); // not necessary for Jacobi
          smoother = mg_smoother;
        }
      else if (settings.smoother=="ssor")
        {
          typedef SSORSmoother Smoother;
          auto *mg_smoother = new MGSmootherPrecondition<MatrixType, Smoother, VectorType>();
          mg_smoother->initialize(mg_matrix, settings.smoother_dampen);
          mg_smoother->set_steps(settings.smoother_steps);
          mg_smoother->set_debug(0);
          mg_smoother->set_symmetric(true);
          smoother = mg_smoother;
        }
#ifdef DEAL_II_WITH_TRILINOS
      else if (settings.smoother=="block jacobi")
        {
          pcout << "block size " << block_size << std::endl;
          typedef BlockJacobiSmoother Smoother;
          auto *mg_smoother = new MGSmootherPrecondition<MatrixType, Smoother, VectorType>();
          mg_smoother->initialize(mg_matrix, Smoother::AdditionalData(
                                    block_size,
                                    "linear",
                                    settings.smoother_dampen));
          mg_smoother->set_steps(1);
          mg_smoother->set_debug(0);
          mg_smoother->set_symmetric(false);
          smoother = mg_smoother;
        }
      else if (settings.smoother=="block ssor")
        {
          pcout << "block size " << block_size << std::endl;
          typedef BlockSSORSmoother Smoother;
          auto *mg_smoother = new MGSmootherPrecondition<MatrixType, Smoother, VectorType>();
          mg_smoother->initialize(mg_matrix, Smoother::AdditionalData(
                                    block_size,
                                    "linear",
                                    settings.smoother_dampen));
          mg_smoother->set_steps(1);
          mg_smoother->set_debug(0);
          mg_smoother->set_symmetric(false);
          smoother = mg_smoother;
        }
#endif
      else
        Assert(false, ExcNotImplemented());

      mg::Matrix<VectorType> mg_m(mg_matrix);
      mg::Matrix<VectorType> mg_down(mg_matrix_dg_down);
      mg::Matrix<VectorType> mg_up(mg_matrix_dg_up);
      mg::Matrix<VectorType> mg_in (mg_interface_in);
      mg::Matrix<VectorType> mg_out(mg_interface_in);

      {
        Multigrid<VectorType > mg(mg_m,
                                  *coarse_grid_solver,
                                  mg_transfer,
                                  *smoother,
                                  *smoother);
        if (!fe->conforms(FiniteElementData<dim>::H1))
          mg.set_edge_flux_matrices(mg_down, mg_up);
        mg.set_edge_matrices(mg_out, mg_in);


        PreconditionMG<dim, VectorType, MGTransferPrebuilt<VectorType> >
        preconditioner(dof_handler, mg, mg_transfer);

        computing_timer.exit_section("Solve: MB Prec Setup");


        {
          TimingStat ts;
          Timer timer(mpi_communicator, true);

          for (unsigned int c=0; c<n_timings; ++c)
            {
              solution = 0.;
              timer.reset();
              timer.start();
              preconditioner.vmult(solution, right_hand_side);
              timer.stop();
              ts.times.push_back(timer.wall_time());
            }

          pcout << "TS prec-vmult: ";
          ts.print(pcout);
          stats.emplace_back("prec-vmult-min", ts.min);
          stats.emplace_back("prec-vmult-avg", ts.avg);

          solution = 0.;
        }

        {

          {
            TimerOutput::Scope timing (computing_timer, "Solve: prec-vmult");
            preconditioner.vmult(solution, right_hand_side);
          }
          solution = 0.;
        }

        {
          TimingStat ts;
          Timer timer(mpi_communicator, true);

          for (unsigned int c=0; c<n_timings; ++c)
            {
              solution = 0.;
              timer.reset();
              timer.start();
              solver.solve (system_matrix, solution, right_hand_side, preconditioner);
              timer.stop();
              ts.times.push_back(timer.wall_time());
            }

          pcout << "TS solve: ";
          ts.print(pcout);
          stats.emplace_back("solve-min", ts.min);
          stats.emplace_back("solve-avg", ts.avg);

          solution = 0.;
        }

//        std::vector<Timer> timer_presmoothing(triangulation.n_global_levels());

//        auto handler_ps = [&](const bool before, const unsigned int level)
//        {
//          if (before) timer_presmoothing[level].start();
//          else timer_presmoothing[level].stop();
//        };

//        mg.connect_pre_smoother_step(handler_ps);
        {
          TimerOutput::Scope timing (computing_timer, "Solve: CG");
          solver.solve(system_matrix, solution, right_hand_side, preconditioner);
        }

        /*  for (unsigned int l=0;l<triangulation.n_global_levels();++l)
            std::cout << "Time pre: " << l
                << "\twall " << timer_presmoothing[l].wall_time()
                << "\tCPU  " << timer_presmoothing[l].cpu_time()
                << std::endl;*/
        constraints.distribute (solution);
      }
      delete smoother;
    }

  double rate = solver_control.final_reduction();
  {
    double r0 = right_hand_side.l2_norm();
    double rn = solver_control.last_value();
    rate = 1.0/solver_control.last_step()*log(r0/rn)/log(10);
  }

  stats.emplace_back("iters", static_cast<double>(solver_control.last_step()));
  stats.emplace_back("rate", rate);

  pcout << "    CG iterations: " << solver_control.last_step()
        << " iters: " << 10.0/rate
        << " rate: " << rate
        << " levels: " << triangulation.n_global_levels()
        << " dofs: " << dof_handler.n_dofs()
        << " assembler: " << settings.assembler_text
        << " smoother_steps: " << settings.smoother_steps
        << " smoother_dampen: " << settings.smoother_dampen;
  if (settings.assembler!=Settings::amg)
    pcout << " smoother: " << settings.smoother;
  pcout << std::endl;
}



template <int dim>
void LaplaceProblem<dim>::estimate ()
{
  TimerOutput::Scope timing (computing_timer, "Estimate");

  VectorType temp_solution;
#ifdef DEAL_II_WITH_TRILINOS
  temp_solution.reinit(locally_relevant_set, MPI_COMM_WORLD);
#else
  temp_solution.reinit(dof_handler.n_dofs());
#endif
  temp_solution = solution;

  estimate_vector.block(0).reinit(triangulation.n_active_cells());
  estimate_vector.collect_sizes();

  std::vector<unsigned int> old_user_indices;
  triangulation.save_user_indices(old_user_indices);

  for (const auto &cell : triangulation.active_cell_iterators())
    cell->set_user_index(cell->active_cell_index());

  // This starts like before,
  MeshWorker::IntegrationInfoBox<dim> info_box;
  const unsigned int                  n_gauss_points =
    dof_handler.get_fe().tensor_degree() + 1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
                                       n_gauss_points + 1,
                                       n_gauss_points);

  AnyData solution_data;
  solution_data.add<VectorType *>(&temp_solution, "solution");

  info_box.cell_selector.add("solution", false, false, true);

  info_box.boundary_selector.add("solution", true, true, false);
  info_box.face_selector.add("solution", true, true, false);

  info_box.add_update_flags_boundary(update_quadrature_points);
  info_box.initialize(*fe, mapping, solution_data, solution);

  MeshWorker::DoFInfo<dim> dof_info(dof_handler);

  MeshWorker::Assembler::CellsAndFaces<double> assembler;
  AnyData                                      out_data;
  out_data.add<BlockVector<double>*>(&estimate_vector, "cells");
  assembler.initialize(out_data, false);

  Estimator<dim> integrator;
  MeshWorker::LoopControl lc;
  lc.faces_to_ghost=MeshWorker::LoopControl::both;
  MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(),
                                         dof_handler.end(),
                                         dof_info,
                                         info_box,
                                         integrator,
                                         assembler,
                                         lc);

  triangulation.load_user_indices(old_user_indices);
}

template <int dim>
void LaplaceProblem<dim>::refine_grid ()
{
  TimerOutput::Scope timing (computing_timer, "Refine grid");

  if (settings.refinement_type=="kelly")
    {
      Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

      VectorType temp_solution;
#ifdef DEAL_II_WITH_TRILINOS
      temp_solution.reinit(locally_relevant_set, MPI_COMM_WORLD);
#else
      temp_solution.reinit(dof_handler.n_dofs());
#endif
      temp_solution = solution;

      KellyErrorEstimator<dim>::estimate (static_cast<DoFHandler<dim>&>(dof_handler),
                                          QGauss<dim-1>(degree+2),
                                          typename FunctionMap<dim>::type(),
                                          temp_solution,
                                          estimated_error_per_cell);
#ifdef DEAL_II_WITH_TRILINOS
      parallel::distributed::GridRefinement::
#else
      GridRefinement::
#endif
      refine_and_coarsen_fixed_fraction (triangulation,
                                         estimated_error_per_cell,
                                         0.5, 0.0);
    }
  else if (settings.refinement_type == "global")
    {
      triangulation.set_all_refine_flags();
    }
  else if (settings.refinement_type == "circle")
    {
      bool flag_set = false;
      typename Triangulation<dim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
      for (; cell != endc; ++cell)
        for (unsigned int vertex=0;
             vertex < GeometryInfo<dim>::vertices_per_cell;
             ++vertex)
          {
            {
              const Point<dim> p = cell->vertex(vertex);
              const Point<dim> origin = (dim == 2 ?
                                         Point<dim>(0,0) :
                                         Point<dim>(0,0,0));
              const double dist = p.distance(origin);
              if (dist<0.25/M_PI)
                {
                  cell->set_refine_flag ();
                  flag_set = true;
                  break;
                }
            }
          }
      if (Utilities::MPI::max(flag_set?1:0, MPI_COMM_WORLD)==0)
        triangulation.set_all_refine_flags();
    }
  else if (settings.refinement_type == "first quadrant")
    {
      typename Triangulation<dim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();

      for (; cell!=endc; ++cell)
        {
          bool first_quadrant = true;
          for (int d=0; d<dim; ++d)
            if (cell->center()[d]>0.0)
              first_quadrant = false;
          if (!first_quadrant)
            continue;

          cell->set_refine_flag();
        }
    }
  else if (settings.refinement_type == "estimator")
    {
#ifdef DEAL_II_WITH_TRILINOS
      parallel::distributed::GridRefinement::
#else
      GridRefinement::
#endif
      refine_and_coarsen_fixed_number (triangulation,
                                       estimate_vector.block(0),
                                       1./(std::pow(2.0,dim)-1.),
                                       0.0);

    }
  else
    throw ExcNotImplemented();

  {
    TimerOutput::Scope timing (computing_timer, "Refine grid: Execute");
    triangulation.execute_coarsening_and_refinement ();
  }
}



template <int dim>
void LaplaceProblem<dim>::output_results (const unsigned int cycle)
{
  TimerOutput::Scope timing (computing_timer, "Output results");

  DataOut<dim> data_out;

  VectorType temp_solution;
#ifdef DEAL_II_WITH_TRILINOS
  temp_solution.reinit(locally_relevant_set, MPI_COMM_WORLD);
#else
  temp_solution.reinit(dof_handler.n_dofs());
#endif
  temp_solution = solution;

  VectorType temp = solution;
  if (settings.assembler == Settings::matrix_free)
    {
      MatrixFreeSystemVector temp_mf;
      MatrixFreeSystemVector solution_copy;
      MatrixFreeSystemVector right_hand_side_copy;
      mf_system_matrix.initialize_dof_vector(temp_mf);
      mf_system_matrix.initialize_dof_vector(solution_copy);
      mf_system_matrix.initialize_dof_vector(right_hand_side_copy);

      ChangeVectorTypes::copy(solution_copy,solution);
      ChangeVectorTypes::copy(right_hand_side_copy,right_hand_side);

      mf_system_matrix.vmult(temp_mf,solution_copy);
      temp_mf.sadd(-1.0,1.0,right_hand_side_copy);

      temp_mf.update_ghost_values();
      ChangeVectorTypes::copy(temp,temp_mf);
    }
  else
    {
      system_matrix.residual(temp,solution,right_hand_side);
    }
  VectorType res_ghosted = temp_solution;
  res_ghosted = temp;




  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (temp_solution, "solution");
  data_out.add_data_vector (res_ghosted, "res");
  Vector<float> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");

  Vector<float> level (triangulation.n_active_cells());
  for (const auto &cell: triangulation.active_cell_iterators())
    level(cell->active_cell_index()) = cell->level();
  data_out.add_data_vector (level, "level");

  if (estimate_vector.size()>0)
    data_out.add_data_vector (estimate_vector, "estimator");

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

      const std::string
      visit_master_filename = ("solution-" +
                               Utilities::int_to_string (cycle, 5) +
                               ".visit");
      std::ofstream visit_master (visit_master_filename.c_str());
      DataOutBase::write_visit_record (visit_master, filenames);
    }
}


template <int dim>
void
LaplaceProblem<dim>::run()
{
  pcout << "Element: " << fe->get_name() << std::endl;
  for (unsigned int cycle=0; cycle<settings.n_steps; ++cycle)
    {
      stats.clear();
      computing_timer.reset();
      pcout << "\n* Cycle " << cycle << ':' << std::endl;
      stats.emplace_back("cycle", static_cast<double>(cycle));
      if (cycle > 0)
        refine_grid ();

      const double imbalance
	= (settings.assembler == Settings::amg) ? -1.0 : MGTools::workload_imbalance(triangulation);

      pcout << "Triangulation "
            << triangulation.n_global_active_cells() << " active cells, "
            << triangulation.n_global_levels() << " levels, "
            << "Workload imbalance: " << imbalance
            << std::endl;
      stats.emplace_back("n_active_cells", static_cast<double>(triangulation.n_global_active_cells()));
      stats.emplace_back("n_global_levels", static_cast<double>(triangulation.n_global_levels()));
      stats.emplace_back("imbalance", imbalance);

      setup_system ();
      stats.emplace_back("ndofs", static_cast<double>(dof_handler.n_dofs()));

      pcout << "DoFHandler " << dof_handler.n_dofs() << " dofs, level dofs";
      if (settings.assembler != Settings::amg)
        {
          for (unsigned int l=0; l<triangulation.n_global_levels(); ++l)
            pcout << ' ' << dof_handler.n_dofs(l);
        }
      pcout << std::endl;

      if (settings.assembler == Settings::matrix_free)
        {
          assemble_rhs_for_matrix_free();
        }
      else
        {
          assemble_system ();
          if (settings.assembler==Settings::matrix_based)
            {
              assemble_multigrid ();
            }
        }

      solve ();
      estimate ();
      if (settings.output)
        output_results (cycle);

      computing_timer.print_summary();
      pcout << ">>";
      pcout << std::setprecision(15);
      for (const auto& stat: stats)
        {
          pcout << " " << stat.first << ": " << stat.second;
        }
      pcout << std::endl;
    }
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  using namespace dealii;

  //std::ofstream logstream;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      //logstream.open("deallog");
      //deallog.attach(logstream);
      deallog.depth_console (3);
    }
  else
    deallog.depth_console (0);

  Settings settings;
  if (!settings.try_parse((argc>1) ? (argv[1]) : ""))
    return 0;

  try
    {
      if (settings.dimension==2)
        {
          LaplaceProblem<2> test(settings);
          test.run();
        }
      else if (settings.dimension==3)
        {
          LaplaceProblem<3> test(settings);
          test.run();
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
      MPI_Abort(MPI_COMM_WORLD, 1);
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
      MPI_Abort(MPI_COMM_WORLD, 2);
      return 1;
    }

  return 0;
}
