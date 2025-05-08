//
//   Project Name:        Kratos
//   Last modified by:    $Author: janosch $
//   Date:                $Date: 2009-01-14 09:40:45 $
//   Revision:            $Revision: 1.2 $
//
//


// System includes

// External includes
#include <boost/python.hpp>


// Project includes
#include "includes/define.h"
#include "python/add_equation_systems_to_python.h"
#include "spaces/ublas_space.h"

#ifdef _OPENMP
#include "spaces/parallel_ublas_space.h"
#endif
#include "linear_solvers/direct_solver.h"
#include "linear_solvers/iterative_solver.h"
#include "external_includes/mkl_pardiso_solver.h"
// #include "external_includes/mkl_complex_pardiso_solver.h"
#include "external_includes/mkl_repeated_pardiso_solver.h"
#include "external_includes/mkl_gmres_solver.h"


namespace Kratos
{

namespace Python
{

void AddLinearSolversToPython()
{
    typedef UblasSpace<KRATOS_DOUBLE_TYPE, CompressedMatrix, Vector> SpaceType;
    typedef UblasSpace<KRATOS_DOUBLE_TYPE, Matrix, Vector> LocalSpaceType;

    typedef LinearSolver<SpaceType, LocalSpaceType, ModelPart> LinearSolverType;
    typedef DirectSolver<SpaceType, LocalSpaceType, ModelPart> DirectSolverType;
    typedef MKLPardisoSolver<SpaceType, LocalSpaceType, ModelPart> MKLPardisoSolverType;
    typedef MKLRepeatedPardisoSolver<SpaceType, LocalSpaceType, ModelPart> MKLRepeatedPardisoSolverType;
    typedef MKLGMRESSolver<SpaceType, LocalSpaceType, ModelPart> MKLGMRESSolverType;
    typedef IterativeSolver<SpaceType, LocalSpaceType, ModelPart> IterativeSolverType;
    typedef Preconditioner<SpaceType, LocalSpaceType, ModelPart> PreconditionerType;

#ifdef _OPENMP
    typedef ParallelUblasSpace<KRATOS_DOUBLE_TYPE, CompressedMatrix, Vector> ParallelSpaceType;
    typedef UblasSpace<KRATOS_DOUBLE_TYPE, Matrix, Vector> ParallelLocalSpaceType;
    typedef LinearSolver<ParallelSpaceType, ParallelLocalSpaceType, ModelPart> ParallelLinearSolverType;
    typedef Reorderer<ParallelSpaceType, ParallelLocalSpaceType > ParallelReordererType;
    typedef DirectSolver<ParallelSpaceType, ParallelLocalSpaceType, ModelPart, ParallelReordererType > ParallelDirectSolverType;
    typedef MKLPardisoSolver<ParallelSpaceType, ParallelLocalSpaceType, ModelPart> ParallelMKLPardisoSolverType;
    typedef MKLGMRESSolver<ParallelSpaceType, ParallelLocalSpaceType, ModelPart> ParallelMKLGMRESSolverType;
    typedef IterativeSolver<ParallelSpaceType, ParallelLocalSpaceType, ModelPart> ParallelIterativeSolverType;
#endif

    using namespace boost::python;

    //***************************************************************************
    //linear solvers
    //***************************************************************************

    class_<MKLPardisoSolverType, MKLPardisoSolverType::Pointer,
           bases<DirectSolverType> >( "MKLPardisoSolver" )
           .def(init<unsigned int>() )
           .def("AdditionalPhysicalDataIsNeeded", &MKLPardisoSolverType::AdditionalPhysicalDataIsNeeded)
           .def("ProvideAdditionalData", &MKLPardisoSolverType::ProvideAdditionalData)
           .def("SetOutOfCore", &MKLPardisoSolverType::SetOutOfCore)
           .def("SetNumThreads", &MKLPardisoSolverType::SetNumThreads)
           .def("SetReordering", &MKLPardisoSolverType::SetReordering)
           .def("SetMessageLevel", &MKLPardisoSolverType::SetMessageLevel)
           ;

    class_<MKLRepeatedPardisoSolverType, MKLRepeatedPardisoSolverType::Pointer,
           bases<DirectSolverType> >( "MKLRepeatedPardisoSolver" )
           .def(init<unsigned int>() )
           .def("SetAdditionalPhysicalData", &MKLRepeatedPardisoSolverType::SetAdditionalPhysicalData)
           .def("AdditionalPhysicalDataIsNeeded", &MKLRepeatedPardisoSolverType::AdditionalPhysicalDataIsNeeded)
           .def("ProvideAdditionalData", &MKLRepeatedPardisoSolverType::ProvideAdditionalData)
           .def("SetOutOfCore", &MKLRepeatedPardisoSolverType::SetOutOfCore)
           ;

#ifdef _OPENMP
    class_<ParallelMKLPardisoSolverType, ParallelMKLPardisoSolverType::Pointer,
           bases<ParallelDirectSolverType> >( "ParallelMKLPardisoSolver" )
           .def(init<unsigned int>() )
           ;
#endif

    class_<MKLGMRESSolverType, MKLGMRESSolverType::Pointer,
           bases<DirectSolverType> >( "MKLGMRESSolver" )
           .def(init<>() )
           ;

#ifdef _OPENMP
    class_<ParallelMKLGMRESSolverType, ParallelMKLGMRESSolverType::Pointer,
           bases<ParallelDirectSolverType> >( "ParallelMKLGMRESSolver" )
           .def(init<>() )
           ;
#endif
}

void AddComplexLinearSolversToPython()
{
    typedef UblasSpace<KRATOS_COMPLEX_TYPE, ComplexCompressedMatrix, ComplexVector> SpaceType;
    typedef UblasSpace<KRATOS_COMPLEX_TYPE, ComplexMatrix, ComplexVector> LocalSpaceType;

    typedef DirectSolver<SpaceType, LocalSpaceType, ComplexModelPart> DirectSolverType;
    typedef MKLPardisoSolver<SpaceType, LocalSpaceType, ComplexModelPart> MKLPardisoSolverType;
    typedef IterativeSolver<SpaceType, LocalSpaceType, ComplexModelPart> IterativeSolverType;
    typedef Preconditioner<SpaceType, LocalSpaceType, ComplexModelPart> PreconditionerType;

    using namespace boost::python;

    //***************************************************************************
    //linear solvers
    //***************************************************************************

    class_<MKLPardisoSolverType, MKLPardisoSolverType::Pointer,
           bases<DirectSolverType> >( "MKLComplexPardisoSolver" )
           .def(init<unsigned int>() )
           .def("AdditionalPhysicalDataIsNeeded", &MKLPardisoSolverType::AdditionalPhysicalDataIsNeeded)
           .def("ProvideAdditionalData", &MKLPardisoSolverType::ProvideAdditionalData)
           .def("SetOutOfCore", &MKLPardisoSolverType::SetOutOfCore)
           .def("SetNumThreads", &MKLPardisoSolverType::SetNumThreads)
           .def("SetReordering", &MKLPardisoSolverType::SetReordering)
           .def("SetMessageLevel", &MKLPardisoSolverType::SetMessageLevel)
           ;
}

void AddGComplexLinearSolversToPython()
{
    typedef UblasSpace<KRATOS_COMPLEX_TYPE, ComplexCompressedMatrix, ComplexVector> SpaceType;
    typedef UblasSpace<KRATOS_COMPLEX_TYPE, ComplexMatrix, ComplexVector> LocalSpaceType;

    typedef DirectSolver<SpaceType, LocalSpaceType, GComplexModelPart> DirectSolverType;
    typedef MKLPardisoSolver<SpaceType, LocalSpaceType, GComplexModelPart> MKLPardisoSolverType;
    typedef IterativeSolver<SpaceType, LocalSpaceType, GComplexModelPart> IterativeSolverType;
    typedef Preconditioner<SpaceType, LocalSpaceType, GComplexModelPart> PreconditionerType;

    using namespace boost::python;

    //***************************************************************************
    //linear solvers
    //***************************************************************************

    class_<MKLPardisoSolverType, MKLPardisoSolverType::Pointer,
           bases<DirectSolverType> >( "MKLGComplexPardisoSolver" )
           .def(init<unsigned int>() )
           .def("AdditionalPhysicalDataIsNeeded", &MKLPardisoSolverType::AdditionalPhysicalDataIsNeeded)
           .def("ProvideAdditionalData", &MKLPardisoSolverType::ProvideAdditionalData)
           .def("SetOutOfCore", &MKLPardisoSolverType::SetOutOfCore)
           .def("SetNumThreads", &MKLPardisoSolverType::SetNumThreads)
           .def("SetReordering", &MKLPardisoSolverType::SetReordering)
           .def("SetMessageLevel", &MKLPardisoSolverType::SetMessageLevel)
           ;
}

}  // namespace Python.

} // Namespace Kratos
