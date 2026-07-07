#ifdef DOEXTERNTEMPLATES
#include "linalg.h"

template class Matrix<numType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct>;
template class Matrix<numType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct>;

template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct>;
template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct>;
template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::direct>;

template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::improperDirect,enums::IndexType::direct>;
template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::improperDirect,enums::IndexType::direct>;
template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::selfAdjoint,enums::IndexType::improperDirect,enums::IndexType::direct>;

template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::improperDual>;
template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::improperDual>;
template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::improperDual>;


template class SparseMatrix<numType,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct>;
template class SparseMatrix<numType,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct>;

template class SparseMatrix<complexType,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct> ;
template class SparseMatrix<complexType,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct> ;
template class SparseMatrix<complexType,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::direct> ;

template class SparseMatrix<complexType,enums::MatrixProperties::None,enums::IndexType::improperDirect,enums::IndexType::direct>;
template class SparseMatrix<complexType,enums::MatrixProperties::Hermitian,enums::IndexType::improperDirect,enums::IndexType::direct>;
template class SparseMatrix<complexType,enums::MatrixProperties::selfAdjoint,enums::IndexType::improperDirect,enums::IndexType::direct>;

template class SparseMatrix<complexType,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::improperDual>;
template class SparseMatrix<complexType,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::improperDual>;
template class SparseMatrix<complexType,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::improperDual>;

template class Matrix<complexType,2,2,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct>;

template class Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::invalid>;
template class Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::direct,enums::IndexType::invalid>;
template class Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::invalid>;
template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::eigenValue>;
template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::direct>;
template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Unitary,enums::IndexType::dual,enums::IndexType::eigenValue>;
#endif
