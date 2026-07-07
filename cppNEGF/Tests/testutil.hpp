#ifndef TESTUTIL_HPP
#define TESTUTIL_HPP

#include "logger.h"
#include "linalg.h"


template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline bool matrixNear(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& a, const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& b, double tol)
{
    auto M = (a-b);
    return M.norm() < tol*sqrt(a.rows()*a.cols());
}


inline bool matrixNear(const ComplexMatrix& a, const ComplexMatrix& b)
{
    auto M = (a-b);
    return M.norm() < 1e-12*sqrt(a.rows()*a.cols());
}

inline bool matrixNear(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b)
{
    auto M = (a-b);
    return M.norm() < 1e-12*sqrt(a.rows()*a.cols());
}

//Work around double not a template parameter until c++20
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline auto makeMatrixNearTol(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& dummy, double tol = 1e-12)
{
    return static_cast<std::function<bool(decltype(dummy),decltype(dummy))>>([tol](decltype(dummy) A, decltype(dummy) B)
    {
        return matrixNear(A,B,tol);
    });
}

inline bool matrixNear(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    Eigen::VectorXd M = (a-b);
    return M.norm() < 1e-12*sqrt(a.rows()*a.cols());
}

inline bool matrixNear(const ComplexSparseMatrix& a, const ComplexSparseMatrix& b)
{
    return (a-b).norm() < 1e-12;
}

template<typename T>
inline bool vectorNear(const std::vector<T>& a, const std::vector<T>& b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++)
    {
        T val = a[i];
        T found = b[i];
        if (!(abs(val - found) < 1e-13 || abs((val-found)/val) < 1e-13))
            return false;
    }
    return true;
}

template<typename valType, size_t N>
inline int expect(valType val, valType found, const char (&funcName)[N], std::string message)
{
    if (val == found)
        return 0;
    logger().printf("Test failed in \"%s\" with message: %s\n",funcName,message.c_str());
    logger().logAccurate("Found", found);
    logger().logAccurate("Expected", val);
    return 1;
}

template<typename valType1,typename valType2, size_t N>
inline int expectNear(const valType1& val, const valType2& found, const std::function<bool(const valType1&, const valType2& )>& comparison, const char (&funcName)[N], std::string message)
{
    if (comparison(val,found))
        return 0;
    logger().printf("Test failed in \"%s\" with message: %s\n",funcName,message.c_str());
    logger().logAccurate("Found", found);
    logger().logAccurate("Expected", val);
    return 1;
}

template<typename valType1,typename valType2, size_t N>
inline int expectNear(const valType1& val, const valType2& found,  bool (*comparison)(const valType1&,const valType2&), const char (&funcName)[N], std::string message)
{
    if (comparison(val,found))
        return 0;
    logger().printf("Test failed in \"%s\" with message: %s\n",funcName,message.c_str());
    logger().logAccurate("Found", found);
    logger().logAccurate("Expected", val);
    return 1;
}
template<size_t N>
inline int expectNear(numType val, numType found, const char (&funcName)[N], std::string message)
{
    if (abs(val - found) < 1e-13 || abs((val-found)/val) < 1e-13)
        return 0;
    logger().printf("Test failed in \"%s\" with message: %s\n",funcName,message.c_str());
    logger().logAccurate("Found", found);
    logger().logAccurate("Expected", val);
    return 1;
}

template<size_t N>
inline int expectNear(numType val, numType found, numType tol, const char (&funcName)[N], std::string message)
{
    if ((val - found) < tol || abs((val-found)/val) < tol)
        return 0;
    logger().printf("Test failed in \"%s\" with message: %s\n",funcName,message.c_str());
    logger().logAccurate("Found", found);
    logger().logAccurate("Expected", val);
    return 1;
}





#endif // TESTUTIL_HPP
