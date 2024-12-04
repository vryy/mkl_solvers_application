/*
* =======================================================================*
* kkkk   kkkk  kkkkkkkkkk   kkkkk    kkkkkkkkkk kkkkkkkkkk kkkkkkkkkK    *
* kkkk  kkkk   kkkk   kkkk  kkkkkk   kkkkkkkkkk kkkkkkkkkk kkkkkkkkkK    *
* kkkkkkkkk    kkkk   kkkk  kkkkkkk     kkkk    kkk    kkk  kkkk         *
* kkkkkkkkk    kkkkkkkkkkk  kkkk kkk	kkkk    kkk    kkk    kkkk       *
* kkkk  kkkk   kkkk  kkkk   kkkk kkkk   kkkk    kkk    kkk      kkkk     *
* kkkk   kkkk  kkkk   kkkk  kkkk  kkkk  kkkk    kkkkkkkkkk  kkkkkkkkkk   *
* kkkk    kkkk kkkk    kkkk kkkk   kkkk kkkk    kkkkkkkkkk  kkkkkkkkkk 	 *
*                                                                        *
* krATos: a fREe opEN sOURce CoDE for mULti-pHysIC aDaptIVe SoLVErS,     *
* aN extEnsIBLe OBjeCt oRiEnTEd SOlutION fOR fInITe ELemEnt fORmULatIONs *
* Copyleft by 2003 ciMNe                                                 *
* Copyleft by 2003 originary authors Copyleft by 2003 your name          *
* This library is free software; you can redistribute it and/or modify   *
* it under the terms of the GNU Lesser General Public License as         *
* published by the Free Software Foundation; either version 2.1 of       *
* the License, or any later version.                                     *
*                                                                        *
* This library is distributed in the hope that it will be useful, but    *
* WITHOUT ANY WARRANTY; without even the implied warranty of             *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                   *
* See the GNU Lesser General Public License for more details.            *
*                                                                        *
* You should have received a copy of the GNU Lesser General Public       *
* License along with this library; if not, write to International Centre *
* for Numerical Methods in Engineering (CIMNE),                          *
* Edifici C1 - Campus Nord UPC, Gran Capità s/n, 08034 Barcelona.        *
*                                                                        *
* You can also contact us to the following email address:                *
* kratos@cimne.upc.es                                                    *
* or fax number: +34 93 401 65 17                                        *
*                                                                        *
* Created at Institute for Structural Mechanics                          *
* Ruhr-University Bochum, Germany                                        *
* Last modified by:    $Author: janosch $  				 *
* Date:                $Date: 2009-01-14 09:40:28 $			 *
* Revision:            $Revision: 1.3 $ 				 *
*========================================================================*
* International Center of Numerical Methods in Engineering - CIMNE	 *
* Barcelona - Spain 							 *
*========================================================================*
*/

#if !defined(KRATOS_MKL_PARDISO_SOLVER_H_INCLUDED )
#define  KRATOS_MKL_PARDISO_SOLVER_H_INCLUDED

// #define BOOST_NUMERIC_BINDINGS_SUPERLU_PRINT

// External includes

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <mkl_types.h>
#include <mkl_service.h>
#include <mkl_pardiso.h> // provides definition for PARDISO function

#include <boost/numeric/bindings/traits/sparse_traits.hpp>
#include <boost/numeric/bindings/traits/matrix_traits.hpp>
#include <boost/numeric/bindings/traits/vector_traits.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>

// Project includes
#include "includes/define.h"
#include "utilities/openmp_utils.h"
#include "includes/ublas_interface.h"
#include "linear_solvers/direct_solver.h"

namespace ublas = boost::numeric::ublas;

namespace Kratos
{
template< class TSparseSpaceType, class TDenseSpaceType,
          class TReordererType = Reorderer<TSparseSpaceType, TDenseSpaceType> >
class MKLPardisoSolver : public DirectSolver< TSparseSpaceType,
    TDenseSpaceType, TReordererType>
{
public:
    /**
     * Counted pointer of MKLPardisoSolver
     */
    KRATOS_CLASS_POINTER_DEFINITION( MKLPardisoSolver );

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TReordererType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    /**
     * @param niter number of iterative refinements allowed
     */
    MKLPardisoSolver(unsigned int niter)
    : mRefinements(niter), mEnableOOC(false), mNumThreads(0)
    , mReordering(2), mMessageLevel(0)
    {
        PrintVersion();
    }

    MKLPardisoSolver()
    : mRefinements(0), mEnableOOC(false), mNumThreads(0)
    , mReordering(2), mMessageLevel(0)
    {
        PrintVersion();
    }

    /**
     * Destructor
     */
    virtual ~MKLPardisoSolver() {}

    void SetOutOfCore(bool OOC)
    {
        mEnableOOC = OOC;
        if(mEnableOOC)
            std::cout << "MKL Out-of-core is enable, adjusting MKL_PARDISO_OOC_MAX_CORE_SIZE to allocate memory for the internal array" << std::endl;
    }

    void SetNumThreads(int num_threads)
    {
        mNumThreads = num_threads;
    }

    void SetReordering(int reordering)
    {
        mReordering = reordering;
    }

    void SetMessageLevel(int message_level)
    {
        mMessageLevel = message_level;
    }

    bool AdditionalPhysicalDataIsNeeded() final
    {
        return false;
    }

    void ProvideAdditionalData(
        SparseMatrixType& rA,
        VectorType& rX,
        VectorType& rB,
        typename ModelPart::DofsArrayType& rdof_set,
        ModelPart& r_model_part
    ) final
    {}

    /**
     * Normal solve method.
     * Solves the linear system Ax=b and puts the result on SystemVector& rX.
     * rX is also th initial guess for iterative methods.
     * @param rA. System matrix
     * @param rX. Solution vector.
     * @param rB. Right hand side vector.
     */
    bool Solve(SparseMatrixType& rA, VectorType& rX, VectorType& rB) final
    {
        double start_solver = OpenMPUtils::GetCurrentTime();
        typedef boost::numeric::bindings::traits::sparse_matrix_traits<SparseMatrixType> matraits;
        typedef boost::numeric::bindings::traits::vector_traits<VectorType> mbtraits;
        typedef typename matraits::value_type val_t;

        MKL_INT n = matraits::size1 (rA);
        assert (n == matraits::size2 (rA));
        assert (n == mbtraits::size (rB));
        assert (n == mbtraits::size (rX));

        // set new number of threads if needed
        int old_num_threads = OpenMPUtils::GetNumThreads();
        if (mNumThreads > 0)
            OpenMPUtils::SetNumThreads(mNumThreads);

        /**
         * nonzeros in rA
         */
        double* a = matraits::value_storage(rA);

        /**
         * manual index vector generation
         */
        MKL_INT *index1_vector = new (std::nothrow) MKL_INT[rA.index1_data().size()];
        MKL_INT *index2_vector = new (std::nothrow) MKL_INT[rA.index2_data().size()];
        std::cout << "Size of the problem: " << n << std::endl;
        std::cout << "Size of index1_vector: " << rA.index1_data().size() << std::endl;
        std::cout << "Size of index2_vector: " << rA.index2_data().size() << std::endl;
        for(unsigned int i = 0; i < rA.index1_data().size(); i++ )
            index1_vector[i] = (MKL_INT)(rA.index1_data()[i])+1;
        for(unsigned int i = 0; i < rA.index2_data().size(); i++ )
            index2_vector[i] = (MKL_INT)(rA.index2_data()[i])+1;

        /**
         *  Matrix type flag:
         * 1    real and structurally symmetric
         * 2    real and symmetic positive definite
         * -2   real and symmetric indefinite
         * 3    complex and structurally symmetric
         * 4    complex and Hermitian positive definite
         * -4   complex and Hermitian indefinite
         * 6    complex and symmetic
         * 11   real and nonsymmetric
         * 13   complex and nonsymmetric
         */
        MKL_INT mtype = 11;
        /* RHS and solution vectors. */
        double *b = mbtraits::storage(rB);
        double *x = mbtraits::storage(rX);

        MKL_INT nrhs = 1; /* Number of right hand sides. */
        /* Internal solver memory pointer pt, */
        /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
        /* or void *pt[64] should be OK on both architectures */
        void *pt[64];
        /* Pardiso control parameters. */
        MKL_INT iparm[64];
        MKL_INT maxfct, mnum, phase, error, msglvl;
        /* Auxiliary variables. */
        MKL_INT i;
        double ddum; /* Double dummy */
        MKL_INT idum; /* Integer dummy. */

        /* -------------------------------------------------------------------- */
        /* .. Setup Pardiso control parameters. */
        /* -------------------------------------------------------------------- */
        for (i = 0; i < 64; i++)
        {
            iparm[i] = 0;
        }
        iparm[0] = 1; /* No solver default */
        iparm[1] = mReordering;
        /* Numbers of processors, value of OMP_NUM_THREADS */
//        iparm[2] = OpenMPUtils::GetNumThreads(); //omp_get_max_threads();
        iparm[2] = OpenMPUtils::GetNumThreads(); //omp_get_num_procs();
        std::cout << "Number of threads/procs (for MKL): " << iparm[2] << std::endl;
        if( mRefinements > 0 )
            iparm[3] = 1; /* iterative-direct algorithm */
        else
            iparm[3] = 0; /* no iterative-direct algorithm */
        iparm[4] = 0; /* No user fill-in reducing permutation */
        iparm[5] = 0; /* Write solution into x */
        iparm[6] = 0; /* Not in use */
        iparm[7] = mRefinements; /* Max numbers of iterative refinement steps */
        iparm[8] = 0; /* Not in use */
        iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
        iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
        iparm[11] = 0; /* Not in use */
        iparm[12] = 1; /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
        iparm[13] = 0; /* Output: Number of perturbed pivots */
        iparm[14] = 0; /* Not in use */
        iparm[15] = 0; /* Not in use */
        iparm[16] = 0; /* Not in use */
        iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
        iparm[18] = -1; /* Output: Mflops for LU factorization */
        iparm[19] = 0; /* Output: Numbers of CG Iterations */
        if(mEnableOOC) iparm[59] = 2; /* enable Out-of-core (to store the matrix in disk; can reduce the performance) */
        maxfct = 1; /* Maximum number of numerical factorizations. */
        mnum = 1; /* Which factorization to use. */
        msglvl = mMessageLevel; /* Print statistical information in file */
        error = 0; /* Initialize error flag */
        /* -------------------------------------------------------------------- */
        /* .. Initialize the internal solver memory pointer. This is only */
        /* necessary for the FIRST call of the PARDISO solver. */
        /* -------------------------------------------------------------------- */
        for (i = 0; i < 64; i++)
        {
            pt[i] = 0;
        }

        /* -------------------------------------------------------------------- */
        /* .. Reordering and Symbolic Factorization. This step also allocates */
        /* all memory that is necessary for the factorization. */
        /* -------------------------------------------------------------------- */
        phase = 11;
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, index1_vector, index2_vector, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error);

        if (error != 0)
        {
            KRATOS_ERROR << "ERROR during symbolic factorization: " << error << std::endl
                         << ErrorCheck(error);
        }
        std::cout << "Reordering completed ... " << std::endl;
        printf("  Number of perturbed pivots ...................... IPARM(14) : %d ~ %.2e\n", iparm[13], (double)iparm[13]);
        printf("  Peak memory symbolic factorization .............. IPARM(15) : %.2e KBs\n", (double)iparm[14]);
        printf("  Permanent memory symbolic factorization ......... IPARM(16) : %.2e KBs\n", (double)iparm[15]);
        printf("  Memory numerical factorization and solution ..... IPARM(17) : %.2e KBs\n", (double)iparm[16]);
        printf("  Number nonzeros in factors ...................... IPARM(18) : %.2e\n", (double)iparm[17]);
        printf("  MFlops of factorization ......................... IPARM(19) : %.2e\n", (double)iparm[18]);
        if(mEnableOOC)
            printf("\n  Size of the minimum OOC memory for numerical factorization and solution (IPARM(63)) = %d KBs", iparm[62]);

        /* -------------------------------------------------------------------- */
        /* .. Numerical factorization. */
        /* -------------------------------------------------------------------- */
        phase = 22;
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, index1_vector, index2_vector, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error);
        if (error != 0)
        {
            KRATOS_ERROR << "ERROR during numerical factorization: " << error << std::endl
                         << ErrorCheck(error);
        }
        std::cout << "Factorization completed ... " << std::endl;

        /* -------------------------------------------------------------------- */
        /* .. Back substitution and iterative refinement. */
        /* -------------------------------------------------------------------- */
        phase = 33;
        iparm[7] = 2; /* Max numbers of iterative refinement steps. */
        /* Set right hand side to one. */
        //for (i = 0; i < n; i++) {
        //    b[i] = 1;
        //}
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, index1_vector, index2_vector, &idum, &nrhs,
                 iparm, &msglvl, b, x, &error);
        if (error != 0)
        {
            KRATOS_ERROR << "ERROR during solution: " << error << std::endl
                         << ErrorCheck(error);
        }

        /* -------------------------------------------------------------------- */
        /* .. Termination and release of memory. */
        /* -------------------------------------------------------------------- */
        phase = -1; /* Release internal memory. */
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, &ddum, index1_vector, index2_vector, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error);
        delete [] index1_vector;
        delete [] index2_vector;

        // restore the original number of threads
        if (mNumThreads > 0)
            OpenMPUtils::SetNumThreads(old_num_threads);

        std::cout << "#### SOLVER TIME: " << OpenMPUtils::GetCurrentTime()-start_solver << " ####" << std::endl;
        return true;
    }

    /**
     * Multi solve method for solving a set of linear systems with same coefficient matrix.
     * Solves the linear system Ax=b and puts the result on SystemVector& rX.
     * rX is also th initial guess for iterative methods.
     * @param rA. System matrix
     * @param rX. Solution vector.
     * @param rB. Right hand side vector.
     */
    bool Solve(SparseMatrixType& rA, DenseMatrixType& rX, DenseMatrixType& rB) final
    {
        double start_solver = OpenMPUtils::GetCurrentTime();
        typedef boost::numeric::bindings::traits::sparse_matrix_traits<SparseMatrixType> matraits;
        typedef boost::numeric::bindings::traits::matrix_traits<DenseMatrixType> mbtraits;
        typedef typename matraits::value_type val_t;

        MKL_INT n = matraits::size1 (rA);
        assert (n == matraits::size2 (rA));
        assert (n == mbtraits::size1 (rB));
        assert (n == mbtraits::size1 (rX));

        // set new number of threads if needed
        int old_num_threads = OpenMPUtils::GetNumThreads();
        if (mNumThreads > 0)
            OpenMPUtils::SetNumThreads(mNumThreads);

        /**
         * nonzeros in rA
         */
        double* a = matraits::value_storage(rA);

        /**
         * manual index vector generation
         */
        MKL_INT *index1_vector = new (std::nothrow) MKL_INT[rA.index1_data().size()];
        MKL_INT *index2_vector = new (std::nothrow) MKL_INT[rA.index2_data().size()];
        std::cout << "Size of the problem: " << n << std::endl;
        std::cout << "Size of index1_vector: " << rA.index1_data().size() << std::endl;
        std::cout << "Size of index2_vector: " << rA.index2_data().size() << std::endl;
        for(unsigned int i = 0; i < rA.index1_data().size(); i++ )
            index1_vector[i] = (MKL_INT)(rA.index1_data()[i])+1;
        for(unsigned int i = 0; i < rA.index2_data().size(); i++ )
            index2_vector[i] = (MKL_INT)(rA.index2_data()[i])+1;

        /**
         *  Matrix type flag:
         * 1    real and structurally symmetric
         * 2    real and symmetic positive definite
         * -2   real and symmetric indefinite
         * 3    complex and structurally symmetric
         * 4    complex and Hermitian positive definite
         * -4   complex and Hermitian indefinite
         * 6    complex and symmetic
         * 11   real and nonsymmetric
         * 13   complex and nonsymmetric
         */
        MKL_INT mtype = 11;
        MKL_INT nrhs = mbtraits::size2(rB); /* Number of right hand sides. */

        /* RHS and solution vectors. */
        DenseMatrixType Bt = trans(rB);
        DenseMatrixType Xt = ZeroMatrix(nrhs, n);
        double *b = mbtraits::storage(Bt);
        double *x = mbtraits::storage(Xt);

        // inefficient copy
//        double b[nrhs * n];
//        double x[nrhs * n];
//        for(int i = 0; i < nrhs; ++i)
//        {
//            std::copy(column(rB, i).begin(), column(rB, i).end(), b + i * n);
//        }

        /* Internal solver memory pointer pt, */
        /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
        /* or void *pt[64] should be OK on both architectures */
        void *pt[64];
        /* Pardiso control parameters. */
        MKL_INT iparm[64];
        MKL_INT maxfct, mnum, phase, error, msglvl;
        /* Auxiliary variables. */
        MKL_INT i;
        double ddum; /* Double dummy */
        MKL_INT idum; /* Integer dummy. */

        /* -------------------------------------------------------------------- */
        /* .. Setup Pardiso control parameters. */
        /* -------------------------------------------------------------------- */
        for (i = 0; i < 64; i++)
        {
            iparm[i] = 0;
        }
        iparm[0] = 1; /* No solver default */
        iparm[1] = mReordering;
        /* Numbers of processors, value of OMP_NUM_THREADS */
        iparm[2] = OpenMPUtils::GetNumThreads(); //omp_get_max_threads();
//        iparm[2] = OpenMPUtils::GetNumProcs(); //omp_get_num_procs();
        std::cout << "Number of threads/procs (for MKL): " << iparm[2] << std::endl;
        if( mRefinements > 0 )
            iparm[3] = 1; /* iterative-direct algorithm */
        else
            iparm[3] = 0; /* no iterative-direct algorithm */
        iparm[4] = 0; /* No user fill-in reducing permutation */
        iparm[5] = 0; /* Write solution into x */
        iparm[6] = 0; /* Not in use */
        iparm[7] = mRefinements; /* Max numbers of iterative refinement steps */
        iparm[8] = 0; /* Not in use */
        iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
        iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
        iparm[11] = 0; /* Not in use */
        iparm[12] = 1; /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
        iparm[13] = 0; /* Output: Number of perturbed pivots */
        iparm[14] = 0; /* Not in use */
        iparm[15] = 0; /* Not in use */
        iparm[16] = 0; /* Not in use */
        iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
        iparm[18] = -1; /* Output: Mflops for LU factorization */
        iparm[19] = 0; /* Output: Numbers of CG Iterations */
        if(mEnableOOC) iparm[59] = 2; /* enable Out-of-core (to store the matrix in disk; can reduce the performance) */
        maxfct = 1; /* Maximum number of numerical factorizations. */
        mnum = 1; /* Which factorization to use. */
        msglvl = mMessageLevel; /* Print statistical information in file */
        error = 0; /* Initialize error flag */

        /* -------------------------------------------------------------------- */
        /* .. Initialize the internal solver memory pointer. This is only */
        /* necessary for the FIRST call of the PARDISO solver. */
        /* -------------------------------------------------------------------- */
        for (i = 0; i < 64; i++)
        {
            pt[i] = 0;
        }

        /* -------------------------------------------------------------------- */
        /* .. Reordering and Symbolic Factorization. This step also allocates */
        /* all memory that is necessary for the factorization. */
        /* -------------------------------------------------------------------- */
        phase = 11;
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, index1_vector, index2_vector, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error);

        if (error != 0)
        {
            KRATOS_ERROR << "ERROR during symbolic factorization: " << error << std::endl
                         << ErrorCheck(error);
        }

        std::cout << "Reordering completed ... " << std::endl;
        //printf("\nNumber of nonzeros in factors = %d", iparm[17]);
        //printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
        /* -------------------------------------------------------------------- */
        /* .. Numerical factorization. */
        /* -------------------------------------------------------------------- */
        KRATOS_WATCH(iparm[63]);
        phase = 22;
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, index1_vector, index2_vector, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error);
        if (error != 0)
        {
            KRATOS_ERROR << "ERROR during numerical factorization: " << error << std::endl
                         << ErrorCheck(error);
        }
        std::cout << "Factorization completed ... " << std::endl;

        /* -------------------------------------------------------------------- */
        /* .. Back substitution and iterative refinement. */
        /* -------------------------------------------------------------------- */
        phase = 33;
        iparm[7] = 2; /* Max numbers of iterative refinement steps. */
        /* Set right hand side to one. */
        //for (i = 0; i < n; i++) {
        //    b[i] = 1;
        //}
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, index1_vector, index2_vector, &idum, &nrhs,
                 iparm, &msglvl, b, x, &error);
        if (error != 0)
        {
            KRATOS_ERROR << "ERROR during solution: " << error << std::endl
                         << ErrorCheck(error);
        }

        /* -------------------------------------------------------------------- */
        /* .. Termination and release of memory. */
        /* -------------------------------------------------------------------- */
        phase = -1; /* Release internal memory. */
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, &ddum, index1_vector, index2_vector, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error);
        delete [] index1_vector;
        delete [] index2_vector;

        // inefficient copy
//        for(int i = 0; i < nrhs; ++i)
//        {
//            std::copy(x + i * n, x + (i + 1) * n, column(rX, i).begin());
//        }

        // restore the original number of threads
        if (mNumThreads > 0)
            OpenMPUtils::SetNumThreads(old_num_threads);

        noalias(rX) = trans(Xt);

        std::cout << "#### SOLVER TIME: " << OpenMPUtils::GetCurrentTime()-start_solver << " ####" << std::endl;
        return true;
    }

    /// Turn back information as a string.
    std::string Info() const final
    {
        return "PARDISO solver";
    }

    /**
     * Print information about this object.
     */
    void PrintInfo(std::ostream& rOStream) const final
    {
        rOStream << "PARDISO solver finished.";
    }

    /**
     * Print object's data.
     */
    void PrintData(std::ostream& rOStream) const final
    {
    }

private:

    int mRefinements;
    bool mEnableOOC;
    int mNumThreads; // this variable is used to override the environment setting OMP_NUM_THREADS, but use exclusively for this solver
    int mReordering; // reordering method for iparm[1]
                        /* 0: minimum degree algorithm */
                        /* 2: Fill-in reordering from METIS */
                        /* 3: Fill-in reordering from METIS 5.1 */
                        /* 4: minimum degree algorithm from AMD */
    int mMessageLevel;  // statistical information; 0: no output, 1: output

    std::string ErrorCheck(MKL_INT error) const
    {
        switch(error)
        {
            case -1:
                return "Input inconsistent";
            case -2:
                return "Not enough memory";
            case -3:
                return "Reordering problem";
            case -4:
                return "Zero pivot, numerical factorization or iterative refinement problem";
            case -5:
                return "Unclassified (internal) error";
            case -6:
                return "Reordering failed (matrix types 11, 13 only)";
            case -7:
                return "Diagonal matrix problem";
            case -8:
                return "32-bit integer overflow problem";
            case -9:
                return "Not enough memory for OOC";
            case -10:
                return "Problems with opening OOC temporary files";
            case -11:
                return "Read/write problems with the OOC data file";
            default:
                return "Unknown";
        }
    }

    void PrintVersion() const
    {
        printf("================================================================\n");
        MKLVersion Version;
        mkl_get_version(&Version);
        printf("MKLPardisoSolver is created, MKL Version info:\n");
        printf("-- Major version:           %d\n", Version.MajorVersion);
        printf("-- Minor version:           %d\n", Version.MinorVersion);
        printf("-- Update version:          %d\n", Version.UpdateVersion);
        printf("-- Product status:          %s\n", Version.ProductStatus);
        printf("-- Build:                   %s\n", Version.Build);
        printf("-- Platform:                %s\n", Version.Platform);
        printf("-- Processor optimization:  %s\n", Version.Processor);
        printf("================================================================\n");
    }

    /**
     * Assignment operator.
     */
    MKLPardisoSolver& operator=(const MKLPardisoSolver& Other);

    /**
     * Copy constructor.
     */
//             MKLPardisoSolver(const MKLPardisoSolver& Other);

}; // Class MKLPardisoSolver


/**
 * input stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType,class TReordererType>
inline std::istream& operator >> (std::istream& rIStream, MKLPardisoSolver< TSparseSpaceType,
                                  TDenseSpaceType, TReordererType>& rThis)
{
    return rIStream;
}

/**
 * output stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream,
                                  const MKLPardisoSolver<TSparseSpaceType,
                                  TDenseSpaceType, TReordererType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}


}  // namespace Kratos.

#endif // KRATOS_MKL_PARDISO_SOLVER_H_INCLUDED  defined
