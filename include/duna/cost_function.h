#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <exception>
#include <Eigen/Dense>
#include "duna/types.h"

namespace duna
{
    /* This class uses CostFunctor to process the total summation cost and linearization
     */

    template <class Scalar = double>
    class CostFunctionBase
    {
    public:

        CostFunctionBase() = delete;

        CostFunctionBase(int num_residuals, int num_model_outputs) : m_num_residuals(num_residuals), m_num_outputs(num_model_outputs)
        {
        }

        CostFunctionBase(const CostFunctionBase &) = delete;
        CostFunctionBase &operator=(const CostFunctionBase &) = delete;
        virtual ~CostFunctionBase() = default;

        void setNumResiduals(int num_residuals) { m_num_residuals = num_residuals; }
        
        virtual Scalar computeCost(const Scalar *x, bool setup_data = true) = 0;
        virtual Scalar linearize(const Scalar *x, Scalar * hessian, Scalar * b) = 0;

    protected:        
        int m_num_residuals;
        int m_num_outputs;
    };
}

#endif