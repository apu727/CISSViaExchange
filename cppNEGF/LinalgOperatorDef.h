/*
 * This file contains the implementations of the operators declared in #include "LinalgOperators.h"
 * It should never be included directly
 * It must come after the definition of Matrix and SparseMatric
 */
#ifndef LINALGOPERATORDEF_H
#define LINALGOPERATORDEF_H

//Allows clang to be able to parse the file
// #define DEBUG_TEMPLATES_DEFS
#include "global.h"
#include "enums.h"

#ifdef DEBUG_TEMPLATES_DEFS
#include "LinalgOperators.h"
#include "global.h"
#endif
inline enums::SpinSymmetry getResultantSpinSymmetry(enums::SpinSymmetry first, enums::SpinSymmetry second)
{
    switch (first)
    {
    case enums::SpinSymmetry::RHF:
        if (second == enums::SpinSymmetry::RHF)
            return enums::SpinSymmetry::RHF;
        else
            return enums::SpinSymmetry::NoSpinSym;
    case enums::SpinSymmetry::NoSpinSym:
        return enums::SpinSymmetry::NoSpinSym;
    case enums::SpinSymmetry::NoSpin:
        releaseAssert(second == enums::SpinSymmetry::NoSpin,"Cannot combine a NOSPIN matrix with a SPIN Matrix");
        return enums::SpinSymmetry::NoSpin;
    case enums::SpinSymmetry::SpinMatrix:
        releaseAssert(second == enums::SpinSymmetry::SpinMatrix,"Cannot combine a SpinMatrix matrix with a not SpinMatrix.. yet");
        return enums::SpinSymmetry::SpinMatrix;
    default:
        releaseAssert(false, "Unhandled case in getResultantSpinSymmetry");
        return enums::SpinSymmetry::NoSpinSym;
    }
}



template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixMultiplication(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                     const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[1]) == BasisManager::buildCanonicalBasisString(B.m_basis[0]) , "Different Basis matrices cannot be multiplied '" + A.m_basis[1] + "', '" + B.m_basis[0] + "'");
    return getResultantSpinSymmetry(A.m_spinSym,B.m_spinSym);
}
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixMultiplication(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[1]) == BasisManager::buildCanonicalBasisString(B.m_basis[0]) , "Different Basis matrices cannot be multiplied '" + A.m_basis[1] + "', '" + B.m_basis[0] + "'");
    return getResultantSpinSymmetry(A.m_spinSym,B.m_spinSym);
}
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixMultiplication(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[1]) == BasisManager::buildCanonicalBasisString(B.m_basis[0]) , "Different Basis matrices cannot be multiplied '" + A.m_basis[1] + "', '" + B.m_basis[0] + "'");
    return getResultantSpinSymmetry(A.m_spinSym,B.m_spinSym);
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixMultiplication(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[1]) == BasisManager::buildCanonicalBasisString(B.m_basis[0]) , "Different Basis matrices cannot be multiplied '" + A.m_basis[1] + "', '" + B.m_basis[0] + "'");
    return getResultantSpinSymmetry(A.m_spinSym,B.m_spinSym);
}

//#########Runtime Addition checks

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixAddition(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                      const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[0]) == BasisManager::buildCanonicalBasisString(B.m_basis[0]) , "Different Basis matrices cannot be Added Part 1'"+ A.m_basis[0] + "', '" + B.m_basis[0] + "'");
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[1]) == BasisManager::buildCanonicalBasisString(B.m_basis[1]) , "Different Basis matrices cannot be Added Part 2'"+ A.m_basis[1] + "', '" + B.m_basis[1] + "'");
    return getResultantSpinSymmetry(A.m_spinSym,B.m_spinSym);
}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixAddition(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                      const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[0]) == BasisManager::buildCanonicalBasisString(B.m_basis[0]) , "Different Basis matrices cannot be Added Part 1'"+ A.m_basis[0] + "', '" + B.m_basis[0] + "'");
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[1]) == BasisManager::buildCanonicalBasisString(B.m_basis[1]) , "Different Basis matrices cannot be Added Part 2'"+ A.m_basis[1] + "', '" + B.m_basis[1] + "'");
    return getResultantSpinSymmetry(A.m_spinSym,B.m_spinSym);
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixAddition(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                      const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[0]) == BasisManager::buildCanonicalBasisString(B.m_basis[0]) , "Different Basis matrices cannot be Added Part 1'"+ A.m_basis[0] + "', '" + B.m_basis[0] + "'");
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[1]) == BasisManager::buildCanonicalBasisString(B.m_basis[1]) , "Different Basis matrices cannot be Added Part 2'"+ A.m_basis[1] + "', '" + B.m_basis[1] + "'");
    return getResultantSpinSymmetry(A.m_spinSym,B.m_spinSym);
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixAddition(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                      const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[0]) == BasisManager::buildCanonicalBasisString(B.m_basis[0]) , "Different Basis matrices cannot be Added Part 1'"+ A.m_basis[0] + "', '" + B.m_basis[0] + "'");
    releaseAssert(BasisManager::buildCanonicalBasisString(A.m_basis[1]) == BasisManager::buildCanonicalBasisString(B.m_basis[1]) , "Different Basis matrices cannot be Added Part 2'"+ A.m_basis[1] + "', '" + B.m_basis[1] + "'");
    return getResultantSpinSymmetry(A.m_spinSym,B.m_spinSym);
}




inline void checkRunTimeTrace(long rows, long cols, const indexBasisT& basis)
{
    releaseAssert(rows == cols,"Cannot trace non square matrix");
    releaseAssert(basis[0] == basis[1],"Basis must be the same for a trace");
}

/////////////////////////////

//Dense Matrix Operators
//Scalar on left
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (T1 scalar, const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat)
{
    Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> ret(mat.m_rows,mat.m_cols,mat.m_basis,mat.m_spinSym);
    ret.m_MatBuffer.get() = mat.m_MatBuffer.get() * scalar;
    return ret;
}


template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (typename Eigen::NumTraits<T1>::Real scalar, const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat)
{
    Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> ret(mat.m_rows,mat.m_cols,mat.m_basis,mat.m_spinSym);
    ret.m_MatBuffer.get() = mat.m_MatBuffer.get() * scalar;
    return ret;
}


//scalar on right
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat,T1 scalar)
{return scalar * mat;}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator/ (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar)
{
    Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> ret(mat.m_rows,mat.m_cols,mat.m_basis,mat.m_spinSym);
    ret.m_MatBuffer.get() = mat.m_MatBuffer.get() / scalar;
    return ret;
}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat,  typename Eigen::NumTraits<T1>::Real scalar)
{return scalar * mat;}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator/ (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, typename Eigen::NumTraits<T1>::Real scalar)
{
    Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> ret(mat.m_rows,mat.m_cols,mat.m_basis,mat.m_spinSym);
    ret.m_MatBuffer.get() = mat.m_MatBuffer.get() / scalar;
    return ret;
}

//*= scalar
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat, T1 scalar)
{
    static_assert(ScalarMultiplicationParams<Params1,T1>::v == Params1 , "Parameter changes under this scalar multiplication so *= not valid");
    mat.m_MatBuffer.get() *= scalar;
    return mat;
}
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar)
{
    static_assert(ScalarMultiplicationParams<Params1,T1>::v == Params1 , "Parameter changes under this scalar multiplication so /= not valid");
    mat.m_MatBuffer.get() /= scalar;
    return mat;
}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat, typename Eigen::NumTraits<T1>::Real scalar)
{//This should always succeed
    static_assert(ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v == Params1 , "Parameter changes under this scalar multiplication so *= not valid");
    mat.m_MatBuffer.get() *= scalar;
    return mat;
}
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, typename Eigen::NumTraits<T1>::Real scalar)
{//This should always succeed
    static_assert(ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v == Params1 , "Parameter changes under this scalar multiplication so /= not valid");
    mat.m_MatBuffer.get() /= scalar;
    return mat;
}
//Sparse Matrix Operators
//Scalar on left
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (T1 scalar, const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat)
{
    SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> ret(mat.m_rows,mat.m_cols,mat.m_basis,mat.m_spinSym);
    ret.m_MatBuffer.get() = mat.m_MatBuffer.get() * scalar;
    return ret;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (typename Eigen::NumTraits<T1>::Real scalar, const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat)
{
    SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> ret(mat.m_rows,mat.m_cols,mat.m_basis,mat.m_spinSym);
    ret.m_MatBuffer.get() = mat.m_MatBuffer.get() * scalar;
    return ret;
}

//scalar on right
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat,T1 scalar)
{return scalar * mat;}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator/ (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar)
{
    SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> ret(mat.m_rows,mat.m_cols,mat.m_basis,mat.m_spinSym);
    ret.m_MatBuffer.get() = mat.m_MatBuffer.get() / scalar;
    return ret;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat,  typename Eigen::NumTraits<T1>::Real scalar)
{return scalar * mat;}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator/ (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat, typename Eigen::NumTraits<T1>::Real scalar)
{
    SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> ret(mat.m_rows,mat.m_cols,mat.m_basis,mat.m_spinSym);
    ret.m_MatBuffer.get() = mat.m_MatBuffer.get() / scalar;
    return ret;
}

//*= scalar
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat, T1 scalar)
{
    static_assert(ScalarMultiplicationParams<Params1,T1>::v == Params1 , "Parameter changes under this scalar multiplication so *= not valid");
    mat.m_MatBuffer.get() *= scalar;
    return mat;
}
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar)
{
    static_assert(ScalarMultiplicationParams<Params1,T1>::v == Params1 , "Parameter changes under this scalar multiplication so /= not valid");
    mat.m_MatBuffer.get() /= scalar;
    return mat;
}
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat, typename Eigen::NumTraits<T1>::Real scalar)
{//This should always succeed
    static_assert(ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v == Params1 , "Parameter changes under this scalar multiplication so *= not valid");
    mat.m_MatBuffer.get() *= scalar;
    return mat;
}
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
inline SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat,typename Eigen::NumTraits<T1>::Real scalar)
{//This should always succeed
    static_assert(ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v == Params1 , "Parameter changes under this scalar multiplication so /= not valid");
    mat.m_MatBuffer.get() /= scalar;
    return mat;
}
//By making the templates generic we can give nice error messages
//##################Matrix Multiplication Operators
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixMultiplication<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixMultiplication(A,B);

    Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,B.m_cols,{A.m_basis[0],B.m_basis[1]},resSym);
    ret.m_MatBuffer->noalias() = A.m_MatBuffer.get() * B.m_MatBuffer.get();
    return ret;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixMultiplication<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixMultiplication(A,B);

    Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,B.m_cols,{A.m_basis[0],B.m_basis[1]},resSym);
    ret.m_MatBuffer->noalias() = A.m_MatBuffer.get() * B.m_MatBuffer.get();
    return ret;
}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixMultiplication<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixMultiplication(A,B);

    Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,B.m_cols,{A.m_basis[0],B.m_basis[1]},resSym);
    ret.m_MatBuffer->noalias() = A.m_MatBuffer.get() * B.m_MatBuffer.get();
    return ret;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixMultiplication<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixMultiplication(A,B);

    SparseMatrix<std::common_type_t<T1,T2>,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,B.m_cols,{A.m_basis[0],B.m_basis[1]},resSym);
    ret.m_MatBuffer.get() = A.m_MatBuffer.get() * B.m_MatBuffer.get();
    return ret;
}

//##################*=, If the Return type is not the same as first then a static_assert fails
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(MatrixMultiplicationParams<Params1,Params2>::v == Params1 , "Parameter changes under this Matrix multiplication so *= not valid");
    //Matrix multiplication needs a temporary anyway so this is fine
    A = std::move(A * B);
    return A;
}

//##################Sparse *= Dense always fails
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B){static_assert(dependent_false<decltype(A)>::value , "Sparse *= Dense is not allowed");}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(MatrixMultiplicationParams<Params1,Params2>::v == Params1 , "Parameter changes under this Matrix multiplication so *= not valid");
    //Matrix multiplication needs a temporary anyway so this is fine
    A = std::move(A * B);
    return A;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(MatrixMultiplicationParams<Params1,Params2>::v == Params1 , "Parameter changes under this Matrix multiplication so *= not valid");
    //Matrix multiplication needs a temporary anyway so this is fine
    A = std::move(A * B);
    return A;
}

//##################Matrix + operators

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixAddition(A,B);

    Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,A.m_cols,A.m_basis,resSym);
    ret.m_MatBuffer.get() = A.m_MatBuffer.get() + B.m_MatBuffer.get();
    return ret;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixAddition(A,B);

    Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,A.m_cols,A.m_basis,resSym);
    ret.m_MatBuffer.get() = A.m_MatBuffer.get() + B.m_MatBuffer.get();
    return ret;
}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixAddition(A,B);

    Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,A.m_cols,A.m_basis,resSym);
    ret.m_MatBuffer.get() = A.m_MatBuffer.get() + B.m_MatBuffer.get();
    return ret;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixAddition(A,B);

    SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,A.m_cols,A.m_basis,resSym);
    ret.m_MatBuffer.get() = A.m_MatBuffer.get() + B.m_MatBuffer.get();
    return ret;
}

//##################Matrix Addition Operators+=, If the Return type is not the same as first then a static_assert fails
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<T1,sizei1,sizej1,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT1>&
operator+=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    static_assert(MatrixAdditionParams<Params1,Params2>::v == Params1 , "Parameter changes under this Matrix addition so += not valid");
    //No aliasing problems here
    A.m_spinSym = checkRunTimeMatrixAddition(A,B);
    A.m_MatBuffer.get() += B.m_MatBuffer.get();
    return A;
}

//Sparse += Dense always fails
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator+=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B){static_assert(dependent_false<decltype(A)>::value , "Sparse += Dense is not allowed");}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator+=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    static_assert(MatrixAdditionParams<Params1,Params2>::v == Params1 , "Parameter changes under this Matrix addition so += not valid");
    //No aliasing problems here
    A.m_spinSym = checkRunTimeMatrixAddition(A,B);
    A.m_MatBuffer.get() += B.m_MatBuffer.get();
    return A;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator+=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    static_assert(MatrixAdditionParams<Params1,Params2>::v == Params1 , "Parameter changes under this Matrix addition so += not valid");
    //No aliasing problems here
    A.m_spinSym = checkRunTimeMatrixAddition(A,B);
    A.m_MatBuffer.get() += B.m_MatBuffer.get();
    return A;
}

//##################Matrix - operators

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixAddition(A,B);
    Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,A.m_cols,A.m_basis,resSym);
    ret.m_MatBuffer.get() = A.m_MatBuffer.get() - B.m_MatBuffer.get();
    return ret;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixAddition(A,B);
    Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,A.m_cols,A.m_basis,resSym);
    ret.m_MatBuffer.get() = A.m_MatBuffer.get() - B.m_MatBuffer.get();
    return ret;
}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixAddition(A,B);
    Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,A.m_cols,A.m_basis,resSym);
    ret.m_MatBuffer.get() = A.m_MatBuffer.get() - B.m_MatBuffer.get();
    return ret;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    enums::SpinSymmetry resSym = checkRunTimeMatrixAddition(A,B);
    SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2> ret(A.m_rows,A.m_cols,A.m_basis,resSym);
    ret.m_MatBuffer.get() = A.m_MatBuffer.get() - B.m_MatBuffer.get();
    return ret;
}

//##################Matrix Multiplication Operators-=, If the Return type is not the same as first then a static_assert fails
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<T1,sizei1,sizej1,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT1>&
operator-=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    static_assert(MatrixAdditionParams<Params1,Params2>::v == Params1 , "Parameter changes under this Matrix addition so -= not valid");
    //No aliasing problems here
    A.m_spinSym = checkRunTimeMatrixAddition(A,B);
    A.m_MatBuffer.get() -= B.m_MatBuffer.get();
    return A;
}

//Sparse -= Dense always fails
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator-=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B){static_assert(dependent_false<decltype(A)>::value , "Sparse -= Dense is not allowed");}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator-=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    static_assert(MatrixAdditionParams<Params1,Params2>::v == Params1 , "Parameter changes under this Matrix addition so -= not valid");
    //No aliasing problems here
    A.m_spinSym = checkRunTimeMatrixAddition(A,B);
    A.m_MatBuffer.get() -= B.m_MatBuffer.get();
    return A;
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator-=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixAddition<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Addition Check Failed");
    static_assert(MatrixAdditionParams<Params1,Params2>::v == Params1 , "Parameter changes under this Matrix addition so -= not valid");
    //No aliasing problems here
    A.m_spinSym = checkRunTimeMatrixAddition(A,B);
    A.m_MatBuffer.get() -= B.m_MatBuffer.get();
    return A;
}

//################## == operators
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
bool
operator==(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    return A.m_MatBuffer.get() == B.m_MatBuffer.get();
}

//##################Combined operators

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Eigen::Vector<std::common_type_t<T1,T2>,sizei1> MulDiagonal(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A, const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixMultiplication<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed");
    static_assert(checkCompileTimeTrace<T1,sizei1,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>::v , "Matrix Trace Check Failed");

    checkRunTimeMatrixMultiplication(A,B);
    checkRunTimeTrace(A.m_rows,B.m_cols,{A.m_basis[0],B.m_basis[1]});

    return (A.m_MatBuffer.get()*B.m_MatBuffer.get()).diagonal().eval();
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Eigen::Vector<std::common_type_t<T1,T2>,sizei2> MulDiagonal(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A, const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixMultiplication<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed");
    static_assert(checkCompileTimeTrace<T1,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>::v , "Matrix Trace Check Failed");

    checkRunTimeMatrixMultiplication(A,B);
    checkRunTimeTrace(A.m_rows,B.m_cols,{A.m_basis[0],B.m_basis[1]});

    return (A.m_MatBuffer.get()*B.m_MatBuffer.get()).diagonal().eval();
}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Eigen::Vector<std::common_type_t<T1,T2>,sizei1> MulDiagonal(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A, const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixMultiplication<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed");
    static_assert(checkCompileTimeTrace<T1,sizei1,Eigen::Dynamic,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>::v , "Matrix Trace Check Failed");

    checkRunTimeMatrixMultiplication(A,B);
    checkRunTimeTrace(A.m_rows,B.m_cols,{A.m_basis[0],B.m_basis[1]});

    return (A.m_MatBuffer.get()*B.m_MatBuffer.get()).diagonal().eval();
}

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Eigen::Vector<std::common_type_t<T1,T2>,Eigen::Dynamic> MulDiagonal(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A, const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixMultiplication<T1,Eigen::Dynamic,Eigen::Dynamic,Params1,m_iIndexT1,m_jIndexT1,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed");
    static_assert(checkCompileTimeTrace<T1,Eigen::Dynamic,Eigen::Dynamic,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>::v , "Matrix Trace Check Failed");

    checkRunTimeMatrixMultiplication(A,B);
    checkRunTimeTrace(A.m_rows,B.m_cols,{A.m_basis[0],B.m_basis[1]});

    return (A.m_MatBuffer.get()*B.m_MatBuffer.get()).diagonal().eval();
}

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2,
         typename T3, int sizei3, int sizej3, enums::MatrixProperties Params3, enums::IndexType m_iIndexT3, enums::IndexType m_jIndexT3>
Matrix<T3,sizei3,sizej3,Params3,m_iIndexT3,m_jIndexT3>&
PEAxB(Matrix<T3,sizei3,sizej3,Params3,m_iIndexT3,m_jIndexT3>& C,
      const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
      const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B)
{
    static_assert(checkCompileTimeMatrixMultiplication<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed");
    enums::SpinSymmetry resSymMult = checkRunTimeMatrixMultiplication(A,B);

    //Extract the resultant type were we to do it
    typedef decltype(A*B) dummy_T;
    static_assert(checkCompileTimeMatrixAddition<T3,sizei3,sizei3,Params1,m_iIndexT3,m_jIndexT3,
              typename dummy_T::static_T, dummy_T::static_sizei, dummy_T::static_sizej,
              dummy_T::static_Params, dummy_T::static_iIndexT, dummy_T::static_jIndexT>::v , "Matrix Addition Check Failed");
    static_assert(MatrixAdditionParams<Params3,dummy_T::static_Params>::v == Params3 , "Parameter changes under this Matrix addition so += not valid");

    dummy_T dummy;
    dummy.m_spinSym = resSymMult;
    dummy.m_basis = {A.m_basis[0],B.m_basis[1]};
    C.m_spinSym = checkRunTimeMatrixAddition(C,dummy);
    typedef typename remove_cvref_t<decltype(C)>::MatType CType;
    typedef typename remove_cvref_t<decltype(A)>::MatType AType;
    typedef typename remove_cvref_t<decltype(B)>::MatType BType;
    releaseAssert((!std::is_same_v<CType,AType> || C.m_MatBuffer.getId() != A.m_MatBuffer.getId()) && (!std::is_same_v<CType,BType> || C.m_MatBuffer.getId() != B.m_MatBuffer.getId()), "Matrices alias in PEAxB, This is not allowed");

    C.m_MatBuffer->noalias() += A.m_MatBuffer.get() * B.m_MatBuffer.get();
    return C;
}
//TODO PEAxBScalar(Scalar, C, A, B);
#endif // LINALGOPERATORDEF_H
