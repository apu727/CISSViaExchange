/*
 * This file contains the forward declarations for Matrix and sparseMatrix. This file allows all the types to be used as pointers
 * Linalg.h should be included in files where the definitions are necessary
 * LinalgForwardDeclarations.h should be included in files where the declarations are necessary
 * Unfortunately Eigen does not provide forward declarations nicely. Therefore we must depend on it
 */

#ifndef LINALGFORWARDDECLARATIONS_H
#define LINALGFORWARDDECLARATIONS_H
#include "global.h"
#include "enums.h"
#include <Eigen/Core>

typedef std::array<std::string,2> indexBasisT;
template <typename T1, enums::MatrixProperties Params1,
         enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
class SparseMatrix;

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1,
         enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
class Matrix;

//Generic Matrices
typedef Matrix<numType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct> RealMatrix;
typedef Matrix<numType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct> RealHermitianMatrix;

typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct> ComplexMatrix;
typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct> ComplexHermitianMatrix;
typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::direct> ComplexSelfAdjointMatrix;

typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::improperDirect,enums::IndexType::direct> ComplexDirectMatrix;
typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::improperDirect,enums::IndexType::direct> ComplexDirectHermitianMatrix;
typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::selfAdjoint,enums::IndexType::improperDirect,enums::IndexType::direct> ComplexDirectSelfAdjointMatrix;

typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::improperDual> ComplexDualMatrix;
typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::improperDual> ComplexDualHermitianMatrix;
typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::improperDual> ComplexDualSelfAdjointMatrix;


typedef SparseMatrix<complexType,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct>  ComplexSparseMatrix;
typedef SparseMatrix<complexType,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct>  ComplexHermitianSparseMatrix;
typedef SparseMatrix<complexType,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::direct>  ComplexSelfAdjointSparseMatrix;

typedef SparseMatrix<complexType,enums::MatrixProperties::None,enums::IndexType::improperDirect,enums::IndexType::direct> ComplexDirectSparseMatrix;
typedef SparseMatrix<complexType,enums::MatrixProperties::Hermitian,enums::IndexType::improperDirect,enums::IndexType::direct> ComplexDirectHermitianSparseMatrix;
typedef SparseMatrix<complexType,enums::MatrixProperties::selfAdjoint,enums::IndexType::improperDirect,enums::IndexType::direct> ComplexDirectSelfAdjointSparseMatrix;

typedef SparseMatrix<complexType,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::improperDual> ComplexDualSparseMatrix;
typedef SparseMatrix<complexType,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::improperDual> ComplexDualHermitianSparseMatrix;
typedef SparseMatrix<complexType,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::improperDual> ComplexDualSelfAdjointSparseMatrix;

//Used for control over Basis functions
typedef ComplexDualHermitianMatrix RaisingMetric;
typedef ComplexDirectHermitianMatrix LoweringMetric;
//These are the same unfortunately but it is still useful to be able to distinguish them
typedef ComplexMatrix ForwardTransformMatrix;
typedef ComplexMatrix InverseTransformMatrix;

//For Spin matrices
typedef Matrix<complexType,2,2,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct> HermitianMatrix2cd;
typedef Matrix<complexType,2,2,enums::MatrixProperties::Unitary,enums::IndexType::dual,enums::IndexType::direct> UnitaryMatrix2cd;

//For Eigen vector things
typedef Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::invalid> ComplexDualVector;
typedef Matrix<complexType,1,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::invalid,enums::IndexType::direct> ComplexDirectVector;
typedef Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::invalid> EigenValueVector;
typedef Matrix<numType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::invalid> RealEigenValueVector;

typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::eigenValue> EigenValueMatrix;
typedef Matrix<numType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::eigenValue,enums::IndexType::eigenValue> RealEigenValueMatrix;

typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::eigenValue> EigenVectorMatrix;
typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::direct> InvEigenVectorMatrix;
typedef Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Unitary,enums::IndexType::dual,enums::IndexType::eigenValue> UnitaryEigenVectorMatrix;

typedef Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::eigenValue> EigenVector;
typedef Matrix<complexType,1,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::direct> InvEigenVector;

typedef Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::Unitary,enums::IndexType::dual,enums::IndexType::eigenValue> UnitaryEigenVector;
typedef Matrix<complexType,1,Eigen::Dynamic,enums::MatrixProperties::Unitary,enums::IndexType::eigenValue,enums::IndexType::direct> UnitaryInvEigenVector;

// typedef ComplexMatrix ComplexMatrixBufferObject; //For compatibility
#endif // LINALGFORWARDDECLARATIONS_H
