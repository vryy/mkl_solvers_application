/*
* =======================================================================*
* kkkk   kkkk  kkkkkkkkkk   kkkkk    kkkkkkkkkk kkkkkkkkkk kkkkkkkkkK    *
* kkkk  kkkk   kkkk   kkkk  kkkkkk   kkkkkkkkkk kkkkkkkkkk kkkkkkkkkK    *
* kkkkkkkkk    kkkk   kkkk  kkkkkkk     kkkk    kkk    kkk  kkkk         *
* kkkkkkkkk    kkkkkkkkkkk  kkkk kkk    kkkk    kkk    kkk    kkkk       *
* kkkk  kkkk   kkkk  kkkk   kkkk kkkk   kkkk    kkk    kkk      kkkk     *
* kkkk   kkkk  kkkk   kkkk  kkkk  kkkk  kkkk    kkkkkkkkkk  kkkkkkkkkk   *
* kkkk    kkkk kkkk    kkkk kkkk   kkkk kkkk    kkkkkkkkkk  kkkkkkkkkk   *
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
* Last modified by:    $Author: janosch $                *
* Date:                $Date: 2009-01-14 09:40:28 $          *
* Revision:            $Revision: 1.3 $                  *
*========================================================================*
* International Center of Numerical Methods in Engineering - CIMNE   *
* Barcelona - Spain                              *
*========================================================================*
*/

#if !defined(KRATOS_MKL_REPEATED_PARDISO_SOLVER_H_INCLUDED )
#define  KRATOS_MKL_REPEATED_PARDISO_SOLVER_H_INCLUDED

// #define BOOST_NUMERIC_BINDINGS_SUPERLU_PRINT

// External includes

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <mkl_types.h>
#include <mkl_pardiso.h>

#include <boost/numeric/bindings/traits/sparse_traits.hpp>
#include <boost/numeric/bindings/traits/matrix_traits.hpp>
#include <boost/numeric/bindings/traits/vector_traits.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>

// Project includes
#include "includes/define.h"
#include "includes/ublas_interface.h"
#include "includes/model_part.h"
#include "linear_solvers/direct_solver.h"
#include "utilities/openmp_utils.h"

namespace ublas = boost::numeric::ublas;

namespace Kratos
{

/*
 * PARDISO solver for initial stiffness strategy
 */
template< class TSparseSpaceType, class TDenseSpaceType,
          class TModelPartType,
          class TReordererType = Reorderer<TSparseSpaceType, TDenseSpaceType> >
class MKLRepeatedPardisoSolver : public DirectSolver< TSparseSpaceType,
    TDenseSpaceType, TModelPartType, TReordererType>
{
public:
    /**
     * Counted pointer of MKLRepeatedPardisoSolver
     */
    KRATOS_CLASS_POINTER_DEFINITION( MKLRepeatedPardisoSolver );

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType, TReordererType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename BaseType::ModelPartType ModelPartType;

    /**
     * @param niter number of iterative refinements allowed
     */
    MKLRepeatedPardisoSolver(unsigned int niter)
    : ma(NULL), mmtype(11), mnrhs(1)
    {
        mRefinements = niter;
        mEnableOOC = false;
        mNeedData = true;
        mIsInitialized = false;
        miparm.resize(64);
        mpt.resize(64);
        mindex1_vector.resize(0);
        mindex2_vector.resize(0);
    }

    MKLRepeatedPardisoSolver()
    : ma(NULL), mmtype(11), mnrhs(1)
    {
        mRefinements = 0;
        mEnableOOC = false;
        mNeedData = true;
        mIsInitialized = false;
        miparm.resize(64);
        mpt.resize(64);
        mindex1_vector.resize(0);
        mindex2_vector.resize(0);
    }

    /**
     * Destructor
     */
    ~MKLRepeatedPardisoSolver() override
    {
        std::cout << "MKLRepeatedPardisoSolver is destroyed" << std::endl;
    }

    void SetOutOfCore(const bool& OOC)
    {
        mEnableOOC = OOC;
        if(mEnableOOC)
            std::cout << "MKL Out-of-core is enable, adjusting MKL_PARDISO_OOC_MAX_CORE_SIZE to allocate memory for the internal array" << std::endl;
    }

    void SetAdditionalPhysicalData(const bool& NeedData)
    {
        mNeedData = NeedData;
    }

    bool AdditionalPhysicalDataIsNeeded() final
    {
        return mNeedData;
    }

    void ProvideAdditionalData(
        SparseMatrixType& rA,
        VectorType& rX,
        VectorType& rB,
        typename ModelPartType::DofsArrayType& rdof_set,
        ModelPartType& r_model_part
    ) final
    {
        typedef boost::numeric::bindings::traits::sparse_matrix_traits<SparseMatrixType> matraits;
        typedef boost::numeric::bindings::traits::vector_traits<VectorType> mbtraits;
        typedef typename matraits::value_type val_t;

        double start_solver = OpenMPUtils::GetCurrentTime();

        MKL_INT n = matraits::size1 (rA);
        assert (n == matraits::size2 (rA));
        assert (n == mbtraits::size (rB));
        assert (n == mbtraits::size (rX));

        // clean the data from previous solve if needed
        if (mIsInitialized)
            this->Clear();

        /**
         * nonzeros in rA
         */
        ma = matraits::value_storage(rA);

        /* Auxiliary variables. */
        MKL_INT phase;
        MKL_INT i;
        double ddum; /* Double dummy */
        MKL_INT idum; /* Integer dummy. */
        /* -------------------------------------------------------------------- */
        /* .. Setup Pardiso control parameters.                                 */
        /* -------------------------------------------------------------------- */
        for (i = 0; i < 64; i++)
        {
            miparm[i] = 0;
        }
        miparm[0] = 1; /* No solver default */
        miparm[1] = 2; /* Fill-in reordering from METIS */
        /* Numbers of processors, value of OMP_NUM_THREADS */
//        iparm[2] = OpenMPUtils::GetNumThreads(); //omp_get_max_threads();
        miparm[2] = OpenMPUtils::GetNumThreads(); //omp_get_num_procs();
        std::cout << "Number of threads/procs (for MKL): " << miparm[2] << std::endl;
        if( mRefinements > 0 )
            miparm[3] = 1; /* iterative-direct algorithm */
        else
            miparm[3] = 0; /* no iterative-direct algorithm */
        miparm[4] = 0; /* No user fill-in reducing permutation */
        miparm[5] = 0; /* Write solution into x */
        miparm[6] = 0; /* Not in use */
        miparm[7] = mRefinements; /* Max numbers of iterative refinement steps */
        miparm[8] = 0; /* Not in use */
        miparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
        miparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
        miparm[11] = 0; /* Not in use */
        miparm[12] = 1; /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
        miparm[13] = 0; /* Output: Number of perturbed pivots */
        miparm[14] = 0; /* Not in use */
        miparm[15] = 0; /* Not in use */
        miparm[16] = 0; /* Not in use */
        miparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
        miparm[18] = -1; /* Output: Mflops for LU factorization */
        miparm[19] = 0; /* Output: Numbers of CG Iterations */
        if(mEnableOOC) miparm[59] = 2; /* enable Out-of-core (to store the matrix in disk; can reduce the performance) */
        mmaxfct = 1; /* Maximum number of numerical factorizations. */
        mmnum = 1; /* Which factorization to use. */
        mmsglvl = 0; /* Print statistical information in file */
        merror = 0; /* Initialize error flag */
        /* -------------------------------------------------------------------- */
        /* .. Initialize the internal solver memory pointer. This is only       */
        /* necessary for the FIRST call of the PARDISO solver.                  */
        /* -------------------------------------------------------------------- */
        for (i = 0; i < 64; i++)
        {
            mpt[i] = 0;
        }
        /**
         * manual index vector generation
         */
        mindex1_vector.resize(rA.index1_data().size());
        mindex2_vector.resize(rA.index2_data().size());
        std::cout << "Size of the problem: " << n << std::endl;
        std::cout << "Size of index1_vector: " << rA.index1_data().size() << std::endl;
        std::cout << "Size of index2_vector: " << rA.index2_data().size() << std::endl;
//                 std::cout << "pardiso_solver: line 156" << std::endl;
        for(unsigned int i = 0; i < rA.index1_data().size(); i++ )
        {
            mindex1_vector[i] = (MKL_INT)(rA.index1_data()[i])+1;
        }
//                 std::cout << "pardiso_solver: line 161" << std::endl;
        for(unsigned int i = 0; i < rA.index2_data().size(); i++ )
        {
            mindex2_vector[i] = (MKL_INT)(rA.index2_data()[i])+1;
        }
        /* -------------------------------------------------------------------- */
        /* .. Reordering and Symbolic Factorization. This step also allocates   */
        /* all memory that is necessary for the factorization.                  */
        /* -------------------------------------------------------------------- */
//                 std::cout << "pardiso_solver: line 241" << std::endl;
        phase = 11;
        PARDISO (mpt.data(), &mmaxfct, &mmnum, &mmtype, &phase,
                 &n, ma, mindex1_vector.data(), mindex2_vector.data(), &idum, &mnrhs,
                 miparm.data(), &mmsglvl, &ddum, &ddum, &merror);

        if (merror != 0)
        {
            std::cout << "ERROR during symbolic factorization: " << merror << std::endl;
            ErrorCheck(merror);
            exit(1);
        }
//                 std::cout << "pardiso_solver: line 251" << std::endl;
        std::cout << "Reordering completed ..." << std::endl;
        printf("  Number of perturbed pivots ...................... IPARM(14) : %d ~ %.2e\n", miparm[13], (double)miparm[13]);
        printf("  Peak memory symbolic factorization .............. IPARM(15) : %.2e KBs\n", (double)miparm[14]);
        printf("  Permanent memory symbolic factorization ......... IPARM(16) : %.2e KBs\n", (double)miparm[15]);
        printf("  Memory numerical factorization and solution ..... IPARM(17) : %.2e KBs\n", (double)miparm[16]);
        printf("  Number nonzeros in factors ...................... IPARM(18) : %.2e\n", (double)miparm[17]);
        printf("  MFlops of factorization ......................... IPARM(19) : %.2e\n", (double)miparm[18]);
        if(mEnableOOC)
            printf("\n  Size of the minimum OOC memory for numerical factorization and solution (IPARM(63)) = %d KBs", miparm[62]);
        /* -------------------------------------------------------------------- */
        KRATOS_WATCH(miparm[63]);

        /* .. Numerical factorization. */
        /* -------------------------------------------------------------------- */
        phase = 22;
        PARDISO (mpt.data(), &mmaxfct, &mmnum, &mmtype, &phase,
                 &n, ma, mindex1_vector.data(), mindex2_vector.data(), &idum, &mnrhs,
                 miparm.data(), &mmsglvl, &ddum, &ddum, &merror);
        if (merror != 0)
        {
            std::cout << "ERROR during numerical factorization: " << merror << std::endl;
            ErrorCheck(merror);
            exit(2);
        }
        std::cout << "Factorization completed ..." << std::endl;

        mIsInitialized = true;

        std::cout << "#### SOLVER PREPARATION TIME: " << OpenMPUtils::GetCurrentTime()-start_solver << " ####" << std::endl;
    }

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
        typedef boost::numeric::bindings::traits::vector_traits<VectorType> mbtraits;

        double start_solver = OpenMPUtils::GetCurrentTime();

        /* RHS and solution vectors. */
        double *b = mbtraits::storage(rB);
        double *x = mbtraits::storage(rX);
        /* -------------------------------------------------------------------- */
        /* .. Back substitution and iterative refinement.                       */
        /* -------------------------------------------------------------------- */
        MKL_INT idum;
        MKL_INT n = mbtraits::size(rB);
        MKL_INT phase = 33;
        miparm[7] = 2; /* Max numbers of iterative refinement steps. */

        PARDISO (mpt.data(), &mmaxfct, &mmnum, &mmtype, &phase,
                 &n, ma, mindex1_vector.data(), mindex2_vector.data(), &idum, &mnrhs,
                 miparm.data(), &mmsglvl, b, x, &merror);
        if (merror != 0)
        {
            std::cout << "ERROR during solution: " << merror << std::endl;
            ErrorCheck(merror);
            exit(3);
        }

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
        typedef boost::numeric::bindings::traits::matrix_traits<DenseMatrixType> mbtraits;

        double start_solver = OpenMPUtils::GetCurrentTime();

        /* RHS and solution vectors. */
        MKL_INT idum;
        MKL_INT n = mbtraits::size1(rB);
        DenseMatrixType Bt = trans(rB);
        DenseMatrixType Xt = ZeroMatrix(mnrhs, n);
        double *b = mbtraits::storage(Bt);
        double *x = mbtraits::storage(Xt);

        /* -------------------------------------------------------------------- */
        /* .. Back substitution and iterative refinement.                       */
        /* -------------------------------------------------------------------- */
        MKL_INT phase = 33;
        miparm[7] = 2; /* Max numbers of iterative refinement steps. */

        PARDISO (mpt.data(), &mmaxfct, &mmnum, &mmtype, &phase,
                 &n, ma, mindex1_vector.data(), mindex2_vector.data(), &idum, &mnrhs,
                 miparm.data(), &mmsglvl, b, x, &merror);
        if (merror != 0)
        {
            std::cout << "ERROR during solution: " << merror << std::endl;
            ErrorCheck(merror);
            exit(3);
        }

        noalias(rX) = trans(Xt);

        std::cout << "#### SOLVER TIME: " << OpenMPUtils::GetCurrentTime()-start_solver << " ####" << std::endl;
        return true;
    }

    void Clear() final
    {
        typedef boost::numeric::bindings::traits::sparse_matrix_traits<SparseMatrixType> matraits;

        if (mIsInitialized)
        {
            /* -------------------------------------------------------------------- */
            /* .. Termination and release of memory.                                */
            /* -------------------------------------------------------------------- */
            MKL_INT phase = -1; /* Release internal memory. */
            MKL_INT n = mindex1_vector.size()-1;
            double ddum; /* Double dummy */
            MKL_INT idum; /* Integer dummy. */
            PARDISO (mpt.data(), &mmaxfct, &mmnum, &mmtype, &phase,
                     &n, &ddum, mindex1_vector.data(), mindex2_vector.data(), &idum, &mnrhs,
                     miparm.data(), &mmsglvl, &ddum, &ddum, &merror);

            mindex1_vector.resize(0);
            mindex2_vector.resize(0);

            mIsInitialized = false;
            std::cout << "#### MKL (REPEATED) PARDISO SOLVER IS FREED ####" << std::endl;
        }
    }

    /// Turn back information as a string.
    std::string Info() const final
    {
        return "Repeated PARDISO solver";
    }

    /**
     * Print information about this object.
     */
    void PrintInfo(std::ostream& rOStream) const final
    {
        rOStream << "Repeated PARDISO solver finished.";
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
    bool mNeedData;
    bool mIsInitialized;

    double* ma;
    std::vector<MKL_INT> mindex1_vector;
    std::vector<MKL_INT> mindex2_vector;

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
    MKL_INT mmtype;

    // number of rhs
    MKL_INT mnrhs;

    /* Internal solver memory pointer pt, */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    std::vector<void*> mpt;
    /* Pardiso control parameters. */
    std::vector<MKL_INT> miparm;
    MKL_INT mmaxfct, mmnum, merror, mmsglvl;

    void ErrorCheck(MKL_INT error)
    {
        switch(error)
        {
            case -1:
                std::cout << "Input inconsistent" << std::endl;
                break;
            case -2:
                std::cout << "Not enough memory" << std::endl;
                break;
            case -3:
                std::cout << "Reordering problem" << std::endl;
                break;
            case -4:
                std::cout << "Zero pivot, numerical factorization or iterative refinement problem" << std::endl;
                break;
            case -5:
                std::cout << "Unclassified (internal) error" << std::endl;
                break;
            case -6:
                std::cout << "Reordering failed (matrix types 11, 13 only)" << std::endl;
                break;
            case -7:
                std::cout << "Diagonal matrix problem" << std::endl;
                break;
            case -8:
                std::cout << "32-bit integer overflow problem" << std::endl;
                break;
            case -9:
                std::cout << "Not enough memory for OOC" << std::endl;
                break;
            case -10:
                std::cout << "Problems with opening OOC temporary files" << std::endl;
                break;
            case -11:
                std::cout << "Read/write problems with the OOC data file" << std::endl;
                break;
        }
    }

    /**
     * Assignment operator.
     */
    MKLRepeatedPardisoSolver& operator=(const MKLRepeatedPardisoSolver& Other);

    /**
     * Copy constructor.
     */
//             MKLRepeatedPardisoSolver(const MKLRepeatedPardisoSolver& Other);

}; // Class MKLRepeatedPardisoSolver

}  // namespace Kratos.

#endif // KRATOS_MKL_REPEATED_PARDISO_SOLVER_H_INCLUDED  defined
