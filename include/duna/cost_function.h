#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <exception>
#include <Eigen/Dense>
#include "duna/types.h"
#include "duna/model.h"
#include <duna/logging.h>


namespace duna
{
    /* This class uses CostFunctor to process the total summation cost and linearization
     */

    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionBase
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using ResidualVector = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, N_PARAMETERS>;

        CostFunctionBase() = default;
        CostFunctionBase(const CostFunctionBase&) = delete;
        CostFunctionBase& operator=(const CostFunctionBase&) = delete;
        virtual ~CostFunctionBase() = default;

        inline virtual void computeAt(const Scalar *x, Scalar *residuals, int index) = 0;
        virtual Scalar computeCost(const Scalar *x) = 0;
        virtual Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b) = 0;

    protected:

    };

    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunction : public CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>
    {
    public:
        using ParameterVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ParameterVector;
        using ResidualVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ResidualVector;
        using HessianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::HessianMatrix;
        using JacobianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::JacobianMatrix;

        // TODO change pointer to smartpointer
        CostFunction(Model<Scalar> * model, int num_residuals) : m_model(model), m_num_residuals(num_residuals), m_num_outputs(N_MODEL_OUTPUTS)
        {
            // m_num_outputs = N_MODEL_OUTPUTS;
            residuals_data = new Scalar[m_num_outputs];
            residuals_plus_data = new Scalar[m_num_outputs];
            if (N_PARAMETERS == -1)
            {
                throw std::runtime_error("Dynamic parameters no yet implemented");
                exit(-1);
            }
        }

        CostFunction(const CostFunction&) = delete;
        CostFunction& operator=(const CostFunction&) = delete;

        ~CostFunction(){
            delete[] residuals_data;
            delete[] residuals_plus_data;
        }

        inline void computeAt(const Scalar *x, Scalar *residuals, int index) override
        {
            m_model->computeAtIndex(x, residuals, index);
        }

        Scalar computeCost(const Scalar *x) override
        {
            Scalar sum = 0;
            
            Eigen::Map<const Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>> residuals(residuals_data);
            
            // TODO convert to class?
            m_model->setup(x);
            for (int i = 0; i < m_num_residuals; ++i)
            {
                computeAt(x, residuals_data, i);                
                sum += residuals.squaredNorm() ;
            }
      
            return sum;
        }

        Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b)
        {
            hessian.setZero();
            b.setZero();

            JacobianMatrix jacobian_row;

            // Map to Eigen            
            Eigen::Map<const ResidualVector> residuals(residuals_data);
            Eigen::Map<const ResidualVector> residuals_plus(residuals_plus_data);

            Scalar sum = 0.0;

            const Scalar epsilon = 24 * (std::numeric_limits<Scalar>::epsilon());

            for (int i = 0; i < m_num_residuals; ++i)
            {
                m_model->setup(x0.data());
                computeAt(x0.data(), residuals_data, i);
                sum += residuals.squaredNorm() ;

                // TODO preallocate functors for each parameter
                for (int j = 0; j < N_PARAMETERS; ++j)
                {
                    ParameterVector x_plus(x0);
                    x_plus[j] += epsilon;

                     // TODO convert to class?
                    m_model->setup(x_plus.data());
                    computeAt(x_plus.data(), residuals_plus_data, i);    
                    jacobian_row.col(j) = (residuals_plus - residuals ) / epsilon;;
                }

                // DUNA_DEBUG_STREAM("JAC:\n" << jacobian_row << "\n");

                hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                b += jacobian_row.transpose() * residuals;
            }
            
            hessian.template triangularView<Eigen::Upper>() = hessian.transpose();
            return sum;
        }

    protected:
        Model<Scalar> *m_model;
         // Holds results for cost computations
        Scalar *residuals_data;
        Scalar *residuals_plus_data;
        const int m_num_residuals;
        const int m_num_outputs;
    };
}
#endif