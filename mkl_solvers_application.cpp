//
//   Project Name:        Kratos
//   Last Modified by:    $Author: janosch $
//   Date:                $Date: 2008-07-23 14:55:54 $
//   Revision:            $Revision: 1.1 $
//
//



// System includes


// External includes


// Project includes
#include "includes/define.h"
#include "mkl_solvers_application.h"


namespace Kratos
{

void KratosMKLSolversApplication::Register()
{

    std::cout << "Initializing KratosMKLSolversApplication..." << std::endl;

}

bool KratosMKLSolversApplication::Has(const std::string& SolverName)
{
    if (SolverName == "MKLPardisoSolver"
     || SolverName == "MKLRepeatedPardisoSolver"
     || SolverName == "MKLComplexPardisoSolver"
     || SolverName == "MKLGComplexPardisoSolver")
    {
#ifdef MKL_ILP64
        return true;
#endif
    }

    return false;
}

}  // namespace Kratos.


