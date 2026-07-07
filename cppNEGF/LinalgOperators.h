/*
 *This file Contains various horrible declarations and definitions to make the linalg templates work. It should never be included directly in calling code.
 * Linalg.h should be included in files where the definitions are necessary
 * LinalgForwardDeclarations.h should be included in files where the declarations are necessary
 */

//Allow clang to be able to parse the file
// #define DEBUG_TEMPLATES

#include "enums.h"
#include "basismanager.h"
#include <type_traits>
#include <Eigen/Core>


#ifdef DEBUG_TEMPLATES
#define LINALG_DECLARE
#define LINALG_FRIEND
#define friend
#endif

#ifdef DEBUG_TEMPLATES_DEFS
#define LINALG_DECLARE
#endif

#ifndef LINALG_HELPER_STRUCT_DEFINES
#define LINALG_HELPER_STRUCT_DEFINES
template <typename T1, enums::MatrixProperties Params1,
         enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
class SparseMatrix;

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1,
         enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
class Matrix;

//This is a C++ 20 feature
template<class T>
struct remove_cvref
{
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template< class T >
using remove_cvref_t = typename remove_cvref<T>::type;

template<enums::IndexType type> struct indexContractionPair{public: static constexpr enums::IndexType v = enums::IndexType::invalid;};
template<> struct indexContractionPair<enums::IndexType::eigenValue>{ public: static constexpr enums::IndexType v = enums::IndexType::eigenValue;};
template<> struct indexContractionPair<enums::IndexType::direct>{ public: static constexpr enums::IndexType v = enums::IndexType::dual;};
template<> struct indexContractionPair<enums::IndexType::dual>{ public: static constexpr enums::IndexType v = enums::IndexType::direct;};
template<> struct indexContractionPair<enums::IndexType::improperDirect>{ public: static constexpr enums::IndexType v = enums::IndexType::improperDual;};
template<> struct indexContractionPair<enums::IndexType::improperDual>{ public: static constexpr enums::IndexType v = enums::IndexType::improperDirect;};

template<enums::IndexType type> struct indexConjugatePair{public: static constexpr enums::IndexType v = enums::IndexType::invalid;};
template<> struct indexConjugatePair<enums::IndexType::eigenValue>{ public: static constexpr enums::IndexType v = enums::IndexType::eigenValue;}; //UNSURE
template<> struct indexConjugatePair<enums::IndexType::direct>{ public: static constexpr enums::IndexType v = enums::IndexType::improperDirect;};
template<> struct indexConjugatePair<enums::IndexType::dual>{ public: static constexpr enums::IndexType v = enums::IndexType::improperDual;};
template<> struct indexConjugatePair<enums::IndexType::improperDirect>{ public: static constexpr enums::IndexType v = enums::IndexType::direct;};
template<> struct indexConjugatePair<enums::IndexType::improperDual>{ public: static constexpr enums::IndexType v = enums::IndexType::dual;};

template<enums::MatrixProperties Params1,enums::MatrixProperties Params2>
struct MatrixMultiplicationParams
{
    static constexpr enums::MatrixProperties v = enums::MatrixProperties::None;
};

template<>
struct MatrixMultiplicationParams<enums::MatrixProperties::Unitary,enums::MatrixProperties::Unitary>
{
    static constexpr enums::MatrixProperties v = enums::MatrixProperties::Unitary;
};

template<enums::MatrixProperties Params1,enums::MatrixProperties Params2>
struct MatrixAdditionParams
{
    static constexpr enums::MatrixProperties v = enums::MatrixProperties::None;
};

template<>
struct MatrixAdditionParams<enums::MatrixProperties::Hermitian,enums::MatrixProperties::Hermitian>
{
    static constexpr enums::MatrixProperties v = enums::MatrixProperties::Hermitian;
};

template<>
struct MatrixAdditionParams<enums::MatrixProperties::selfAdjoint,enums::MatrixProperties::selfAdjoint>
{
    static constexpr enums::MatrixProperties v = enums::MatrixProperties::selfAdjoint;
};

template<enums::MatrixProperties Params1,typename T>
struct ScalarMultiplicationParams
{
    static constexpr enums::MatrixProperties v = std::is_same_v<T,typename Eigen::NumTraits<T>::Real> ? Params1 : enums::MatrixProperties::None;
};

template<enums::MatrixProperties Params1,typename T>
struct TraceReturn
{
    using t = std::conditional_t<Params1 == enums::MatrixProperties::selfAdjoint, typename Eigen::NumTraits<T>::Real,T>;
};

template<typename MatType>
struct EigReturn
{
    static void check()
    {
        static_assert(MatType::static_iIndexT == indexContractionPair<MatType::static_jIndexT>::v , "Eigenvalues and vectors only in a biorthogonal basis. See also Generalised Eigenvalue problem");
    }
    typedef typename MatType::static_T T;
    static constexpr enums::MatrixProperties Params1 = MatType::static_Params;
    typedef typename Eigen::NumTraits<T>::Real realT;
    static constexpr bool TIsReal = std::is_same_v<T,realT>;
    static constexpr bool isSelfAdjoint = Params1 == enums::MatrixProperties::selfAdjoint;
    static constexpr bool isHermitian = Params1 == enums::MatrixProperties::Hermitian;
    typedef std::conditional_t<TIsReal && (isHermitian || isSelfAdjoint), realT,std::complex<realT>> Vec;
    static constexpr enums::MatrixProperties VecParams = isHermitian ? enums::MatrixProperties::Unitary : (isSelfAdjoint? enums::MatrixProperties::NormPreserving : enums::MatrixProperties::None) ;
    typedef std::conditional_t<isSelfAdjoint || isHermitian, realT,std::complex<realT>> Val;
};

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
struct HermReturn //Computes the matrix type that is returned when we try to move to a Hermitian basis
{
    typedef typename Eigen::NumTraits<T1>::Real realT;
    static constexpr bool TIsReal = std::is_same_v<T1,realT>;
    static constexpr bool isSelfAdjoint = Params1 == enums::MatrixProperties::selfAdjoint || Params1 == enums::MatrixProperties::Hermitian;
    static void check()
    {
        static_assert(isSelfAdjoint,"Cannot make non self adjoint matrix Hermitian");
    }

    typedef std::complex<realT> InternalT; // The eigenbasis would be real but in general this is not necessarily real
    typedef Matrix<InternalT,sizei1,sizej1,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct> Mat;
};

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
struct HermReturnSparse //Computes the matrix type that is returned when we try to move to a Hermitian basis
{
    typedef typename Eigen::NumTraits<T1>::Real realT;
    static constexpr bool TIsReal = std::is_same_v<T1,realT>;
    static constexpr bool isSelfAdjoint = Params1 == enums::MatrixProperties::selfAdjoint || Params1 == enums::MatrixProperties::Hermitian;
    static void check()
    {
        static_assert(isSelfAdjoint,"Cannot make non self adjoint matrix Hermitian");
    }

    typedef std::complex<realT> InternalT; // The eigenbasis would be real but in general this is not necessarily real
    typedef SparseMatrix<InternalT,enums::MatrixProperties::Hermitian,m_iIndexT1,m_jIndexT1> Mat;
};

//This is very annoying/impossible(different basis) to know at compile time. For future improvements this struct has been created
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
struct checkBasisReturn //Checks and provides debug if the basis change type is wrong
{
    typedef typename Eigen::NumTraits<T1>::Real realT1;
    static constexpr bool isSelfAdjoint = Params1 == enums::MatrixProperties::selfAdjoint;
    static constexpr bool isNormPreserving = Params1 == enums::MatrixProperties::NormPreserving;

    static_assert(std::is_same_v<std::complex<realT1>,T2> , "Basis change Scalar type is wrong. Likely needs to be complex");
    static_assert((Params2 ==(
                        /*if*/ isSelfAdjoint ?
                            enums::MatrixProperties::selfAdjoint
                       /*else*/:(
                            /*if*/isNormPreserving ?
                                enums::MatrixProperties::NormPreserving
                            /*else*/:
                                enums::MatrixProperties::None)
                        )
                   || Params2 == enums::MatrixProperties::None), "Basis change Parameters are incorrect");
    static constexpr bool v = true;
    //The only requirement to pass this is that it is a complex type
};



//Return type is automatically deduced to not lead to compilation failures
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
struct checkCompileTimeMatrixMultiplication
{
    static_assert(m_jIndexT1 == indexContractionPair<m_iIndexT2>::v , "Matrices cannot be contracted");
    static_assert(sizej1 == sizei2 , "Matrices wrong size for contraction - Matrix Multiplication");
    static constexpr bool v = true;
};

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
struct checkCompileTimeMatrixAddition
{
    static_assert(m_iIndexT1 == m_iIndexT2 , "Matrices cannot be added, differing index type");
    static_assert(m_jIndexT1 == m_jIndexT2 , "Matrices cannot be added, differing index type");
    static_assert(sizei1 == sizei2 && sizej1 == sizej2 , "Matrices wrong size for Matrix Addition");
    static_assert(std::is_same_v<T1,T2>, "Matrix addition between different types is not allowed");
    static constexpr bool v = true;
};

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
struct checkCompileTimeTrace
{
    static_assert(m_iIndexT1 == indexContractionPair<m_jIndexT1>::v , "Matrices cannot be contracted");
    static_assert(sizei1 == sizej1 , "Matrices wrong size for contraction - Trace");
    static constexpr bool v = true;
};

template <typename MatType, typename destType>
struct SpinUnBlocking
{
    typedef typename MatType::static_T T;
    static constexpr int sizei = MatType::static_sizei ;
    static constexpr int sizej = MatType::static_sizej;
    static constexpr enums::MatrixProperties destParams = destType::static_Params;
    static constexpr enums::IndexType m_iIndexT = MatType::static_iIndexT;
    static constexpr enums::IndexType m_jIndexT = MatType::static_jIndexT;


    static constexpr bool isMatrix = sizei == sizej;
    static constexpr bool iIsNotTransformedType = m_iIndexT == enums::IndexType::eigenValue || m_iIndexT == enums::IndexType::invalid;
    static constexpr bool jIsNotTransformedType = m_iIndexT == enums::IndexType::eigenValue || m_jIndexT == enums::IndexType::invalid;
    //Changes size if sizei != 1 and IS__TransformedType type except if sizei == -1 in which case it stays -1.
    //If it is size=1 then it is either a 1x1 matrix. Which cannot be spin blocked so runtime crash or 1xN or Nx1 vector.
    //for a 1x1 matrix we must pass the static asserts. The best way to do this is to claim it doesnt change size.
    //if it is a 1xN or Nx1 vector then the index corresponding to 1 is marked as invalid. This means IsNotTransformedType == True and !IsNotTransformedType == false.
    //So also does not change size
    static constexpr bool iChangesSize = !iIsNotTransformedType && sizei != 1 && sizei != Eigen::Dynamic;
    static constexpr bool jChangesSize = !jIsNotTransformedType && sizej != 1 && sizej != Eigen::Dynamic;

    static constexpr int destsizei = iChangesSize ? 2*sizei : sizei;
    static constexpr int destsizej = jChangesSize ? 2*sizej : sizej;

    //This should always evaluate to true unless the above logic is broken
    // static_assert((destsizei == Eigen::Dynamic && destsizej == Eigen::Dynamic) ||
    //                   (destsizei == Eigen::Dynamic && destsizej == 1 && sizej == 1)  ||
    //                   (destsizej == Eigen::Dynamic && destsizei == 1 && sizei == 1)  ||
    //                   (destsizei == 2*sizei && destsizej == 2*sizej),"Destination matrix has wrong size");
    static_assert(destsizei == Eigen::Dynamic || (iChangesSize && destsizei == 2*sizei) || (!iChangesSize && destsizei == sizei), "Unblocking Logic broken");
    static_assert(destsizej == Eigen::Dynamic || (jChangesSize && destsizei == 2*sizej) || (!jChangesSize && destsizej == sizej), "Unblocking Logic broken");


    typedef Matrix<T,destsizei,destsizej, destParams, destType::static_iIndexT,destType::static_jIndexT> DenseResult;
    typedef SparseMatrix<T, destParams, destType::static_iIndexT,destType::static_jIndexT> SparseResult;


    static void check(const MatType& src,
               const destType& dest, const std::array<BasisManager::possibleTags,2>& tagRemoved, bool silenceUnblockWarning = false)
    {
        if (tagRemoved[0] == BasisManager::possibleTags::NullBasis && tagRemoved[1] == BasisManager::possibleTags::NullBasis)
            return; // not actually doing anything

        enums::SpinSymmetry sym = src.getSpinSym();
        switch(sym)
        {
        case enums::SpinSymmetry::RHF:
            break;
        case enums::SpinSymmetry::NoSpinSym:
            if (!silenceUnblockWarning) logger().log("Unblocking a NoSpinSym matrix, while allowed are you sure this is what you wanted");
            break;
        case enums::SpinSymmetry::NoSpin:
            releaseAssert(false,"Cannot unblock a nospin matrix");
            break;
        case enums::SpinSymmetry::SpinMatrix:
            releaseAssert(false,"Cannot unblock a spin matrix");
            break;
        default:
            releaseAssert(false,"Unhandled switch case");
        }

        if constexpr(destType::static_sizei != Eigen::Dynamic)
            releaseAssert((iChangesSize && destType::static_sizei == 2*src.rows()) || (!iChangesSize && destType::static_sizei == src.rows()),"Dest matrix Wrong isize for Unblocking");

        if constexpr(destType::static_sizej != Eigen::Dynamic)
            releaseAssert((jChangesSize && destType::static_sizej == 2*src.cols()) || (!jChangesSize && destType::static_sizej == src.cols()),"Dest matrix Wrong jsize for Unblocking");
        //Because the validity of the parameters depends on runtime variables (tagRemoved) this cannot be checked at compile time
        if ((tagRemoved[0] == BasisManager::possibleTags::ABlock && tagRemoved[1] == BasisManager::possibleTags::BBlock) || ((tagRemoved[1] == BasisManager::possibleTags::ABlock && tagRemoved[0] == BasisManager::possibleTags::BBlock)))
            releaseAssert(destParams == enums::MatrixProperties::None, "UnBlocking from a AB/BA block cannot have any matrix properties");

    }
};

template <typename MatType, typename destType>
struct SpinBlocking
{
    typedef typename MatType::static_T T;
    static constexpr int sizei = MatType::static_sizei ;
    static constexpr int sizej = MatType::static_sizej;
     static constexpr enums::MatrixProperties destParams = destType::static_Params;
    static constexpr enums::IndexType m_iIndexT = MatType::static_iIndexT;
    static constexpr enums::IndexType m_jIndexT = MatType::static_jIndexT;


    static constexpr bool isMatrix = sizei == sizej;
    static constexpr bool iIsNotTransformedType = m_iIndexT == enums::IndexType::eigenValue || m_iIndexT == enums::IndexType::invalid;
    static constexpr bool jIsNotTransformedType = m_iIndexT == enums::IndexType::eigenValue || m_jIndexT == enums::IndexType::invalid;

    //see Spin unblocking
    static constexpr bool iChangesSize = !iIsNotTransformedType && sizei != 1 && sizei != Eigen::Dynamic;
    static constexpr bool jChangesSize = !jIsNotTransformedType && sizej != 1 && sizej != Eigen::Dynamic;

    static constexpr int destsizei = iChangesSize ? sizei/2 : sizei;
    static constexpr int destsizej = jChangesSize ? sizej/2 : sizej;

    //This should always evaluate to true unless the above logic is broken
    static_assert((destsizei == Eigen::Dynamic) || (iChangesSize && 2*destsizei == sizei) || (!iChangesSize && destsizei == sizei),"Unblocking Logic broken");
    static_assert((destsizej == Eigen::Dynamic) || (jChangesSize && 2*destsizej == sizej) || (!jChangesSize && destsizej == sizej),"Unblocking Logic broken");

    typedef Matrix<T,destsizei,destsizej, destParams, destType::static_iIndexT,destType::static_jIndexT> DenseResult;
    typedef SparseMatrix<T, destParams,  destType::static_iIndexT,destType::static_jIndexT> SparseResult;

    static void check(const MatType& src,
                      const destType& dest, const std::array<BasisManager::possibleTags,2>& tagRemoved)
    {
        if (tagRemoved[0] == BasisManager::possibleTags::NullBasis && tagRemoved[1] == BasisManager::possibleTags::NullBasis)
            return; // not actually doing anything
        //This has to be a runtime assert as the function is instantiated even for objects for which this cant be done
        enums::SpinSymmetry sym = src.getSpinSym();
        switch(sym)
        {
        case enums::SpinSymmetry::RHF:
        case enums::SpinSymmetry::NoSpinSym:
            break;
        case enums::SpinSymmetry::NoSpin:
            releaseAssert(false,"Cannot block a nospin matrix");
            break;
        case enums::SpinSymmetry::SpinMatrix:
            releaseAssert(false,"Cannot block a spin matrix");
            break;
        default:
            releaseAssert(false,"Unhandled switch case");
        }
        if constexpr(destType::static_sizei != Eigen::Dynamic)
        {
            releaseAssert((!iChangesSize || src.rows() %2 == 0),"Source matrix cannot be blocked, Wrong isize");
            releaseAssert((iChangesSize && 2*dest.rows() == destType::static_sizei) || (!iChangesSize && destType::static_sizei == src.rows()),"Dest matrix Wrong isize for blocking");
        }
        if constexpr(destType::static_sizej != Eigen::Dynamic)
        {
            releaseAssert((!jChangesSize || src.cols() %2 == 0),"Source matrix cannot be blocked, Wrong jsize");
            releaseAssert((jChangesSize && 2*destType::static_sizej == src.cols()) || (!jChangesSize && destType::static_sizej == src.cols()),"Dest matrix Wrong jsize for blocking");
        }


        //Because the validity of the parameters depends on runtime variables (tagRemoved) this cannot be checked at compile time
        if ((tagRemoved[0] == BasisManager::possibleTags::ABlock && tagRemoved[1] == BasisManager::possibleTags::BBlock) || ((tagRemoved[1] == BasisManager::possibleTags::ABlock && tagRemoved[0] == BasisManager::possibleTags::BBlock)))
            releaseAssert(destParams == enums::MatrixProperties::None,"Resultant parameters of basis change with spin unblocking wrong, Should be enums::MatrixProperties::None");

    }


};

template <typename MatType,
         typename T1 = typename MatType::static_T, int sizei1 = MatType::static_sizei, int sizej1 = MatType::static_sizej,
         enums::MatrixProperties Params1 = MatType::static_Params, enums::IndexType m_iIndexT1 = MatType::static_iIndexT, enums::IndexType m_jIndexT1 = MatType::static_jIndexT>
struct changeMatrixArgs
{
    typedef Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> Dense;
    typedef SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> Sparse;
};
//Helper to only change the parameters
template <typename MatType,enums::MatrixProperties newParams>
struct changeMatrixParameters
{
    typedef typename changeMatrixArgs<MatType,typename MatType::static_T,MatType::static_sizei,MatType::static_sizej,newParams>::Dense Dense;
    typedef typename changeMatrixArgs<MatType,typename MatType::static_T,MatType::static_sizei,MatType::static_sizej,newParams>::Sparse Sparse;
};

#endif
#ifdef LINALG_DECLARE

inline enums::SpinSymmetry getResultantSpinSymmetry(enums::SpinSymmetry first, enums::SpinSymmetry second);


//Return type is automatically deduced to not lead to compilation failures
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
struct checkCompileTimeMatrixMultiplication;

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
struct checkCompileTimeTrace;

//##Runtime Multiplication checks
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixMultiplication(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixMultiplication(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixMultiplication(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixMultiplication(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##########Runtime Addition checks
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixAddition(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixAddition(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixAddition(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline enums::SpinSymmetry checkRunTimeMatrixAddition(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline void checkRunTimeMatrixSpinUnblocking(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& src,
                                                                 const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& dest, BasisManager::possibleTags tagRemoved);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
inline void checkRunTimeMatrixSpinBlocking(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& src,
                                                                 const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& dest, BasisManager::possibleTags tagRemoved);


inline void checkRunTimeTrace(long rows, long cols, const indexBasisT& basis);

//TODO real matrix times complex scalar?
//Dense Matrix Operators
//Scalar on left
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (T1 scalar, const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat);


template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (typename Eigen::NumTraits<T1>::Real scalar, const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat);


//scalar on right
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat,T1 scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator/ (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat,  typename Eigen::NumTraits<T1>::Real scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator/ (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, typename Eigen::NumTraits<T1>::Real scalar);

//*= scalar
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat, T1 scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat, typename Eigen::NumTraits<T1>::Real scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, typename Eigen::NumTraits<T1>::Real scalar);

//Sparse Matrix Operators
//Scalar on left
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (T1 scalar, const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat);


template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (typename Eigen::NumTraits<T1>::Real scalar, const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat);


//scalar on right
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat,T1 scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator/ (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat,  typename Eigen::NumTraits<T1>::Real scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator/ (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat, typename Eigen::NumTraits<T1>::Real scalar);

//*= scalar
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat, T1 scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
inline SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat, typename Eigen::NumTraits<T1>::Real scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool> = 1>
inline SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat,typename Eigen::NumTraits<T1>::Real scalar);

//By making the templates generic we can give nice error messages
//##################Matrix Multiplication Operators
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################*=, If the Return type is not the same as first then a static_assert fails
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Sparse *= Dense always fails
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Matrix + operators

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Matrix Multiplication Operators+=, If the Return type is not the same as first then a static_assert fails
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<T1,sizei1,sizej1,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT1>&
operator+=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

//Sparse += Dense always fails
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator+=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator+=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator+=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Matrix - operators

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Matrix Multiplication Operators-=, If the Return type is not the same as first then a static_assert fails
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<T1,sizei1,sizej1,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT1>&
operator-=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

//Sparse += Dense always fails
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator-=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator-=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator-=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//################## == operators
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
bool
operator==(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Combined operators
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Eigen::Vector<std::common_type_t<T1,T2>,sizei1> MulDiagonal(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A, const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Eigen::Vector<std::common_type_t<T1,T2>,sizei2> MulDiagonal(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A, const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Eigen::Vector<std::common_type_t<T1,T2>,sizei1> MulDiagonal(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A, const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
Eigen::Vector<std::common_type_t<T1,T2>,Eigen::Dynamic> MulDiagonal(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A, const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
          typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2,
          typename T3, int sizei3, int sizej3, enums::MatrixProperties Params3, enums::IndexType m_iIndexT3, enums::IndexType m_jIndexT3>
Matrix<T3,sizei3,sizej3,Params3,m_iIndexT3,m_jIndexT3>&
PEAxB(Matrix<T3,sizei3,sizej3,Params3,m_iIndexT3,m_jIndexT3>& C,
      const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
      const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);
//Computes C += A*B efficiently
#endif
#ifdef LINALG_FRIEND
//##########Runtime multiplication checks
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend enums::SpinSymmetry checkRunTimeMatrixMultiplication(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend enums::SpinSymmetry checkRunTimeMatrixMultiplication(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend enums::SpinSymmetry checkRunTimeMatrixMultiplication(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend enums::SpinSymmetry checkRunTimeMatrixMultiplication(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                            const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);
//##########Runtime addition checks
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend enums::SpinSymmetry checkRunTimeMatrixAddition(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                      const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend enums::SpinSymmetry checkRunTimeMatrixAddition(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                      const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend enums::SpinSymmetry checkRunTimeMatrixAddition(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                      const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend enums::SpinSymmetry checkRunTimeMatrixAddition(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
                                                      const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

friend void checkRunTimeTrace(long rows, long cols);



template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (T1 scalar, const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat);


template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (typename Eigen::NumTraits<T1>::Real scalar, const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat);


//scalar on right
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat,T1 scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator/ (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat,  typename Eigen::NumTraits<T1>::Real scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend Matrix<T1,sizei1,sizej1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator/ (const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, typename Eigen::NumTraits<T1>::Real scalar);

//*= scalar
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat, T1 scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1> & mat, typename Eigen::NumTraits<T1>::Real scalar);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat, typename Eigen::NumTraits<T1>::Real scalar);

//Sparse Matrix Operators
//Scalar on left
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (T1 scalar, const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (typename Eigen::NumTraits<T1>::Real scalar, const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat);

//scalar on right
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator* (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat,T1 scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend SparseMatrix<T1,ScalarMultiplicationParams<Params1,T1>::v,m_iIndexT1,m_jIndexT1> operator/ (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator* (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat,  typename Eigen::NumTraits<T1>::Real scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend SparseMatrix<T1,ScalarMultiplicationParams<Params1,typename Eigen::NumTraits<T1>::Real>::v,m_iIndexT1,m_jIndexT1> operator/ (const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat, typename Eigen::NumTraits<T1>::Real scalar);

//*= scalar
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat, T1 scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
friend SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat, T1 scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator*= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1> & mat, typename Eigen::NumTraits<T1>::Real scalar);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1, std::enable_if_t<!std::is_same_v<T1,typename Eigen::NumTraits<T1>::Real>,bool>>
friend SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& operator/= (SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat,typename Eigen::NumTraits<T1>::Real scalar);

//By making the templates generic we can give nice error messages
//##################Matrix Multiplication Operators
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend SparseMatrix<std::common_type_t<T1,T2>,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator*(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################*=, If the Return type is not the same as first then a static_assert fails
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Sparse *= Dense always fails
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend SparseMatrix<std::common_type_t<T1,T2>,MatrixMultiplicationParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator*=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Matrix + operators

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator+(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Matrix Multiplication Operators+=, If the Return type is not the same as first then a static_assert fails
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<T1,sizei1,sizej1,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT1>&
operator+=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

//Sparse += Dense always fails
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator+=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator+=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator+=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Matrix - operators

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,sizei1,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>
operator-(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
          const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Matrix Multiplication Operators-=, If the Return type is not the same as first then a static_assert fails
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<T1,sizei1,sizej1,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT1>&
operator-=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

//Sparse += Dense always fails
template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,Eigen::Dynamic,sizej2,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator-=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Matrix<std::common_type_t<T1,T2>,sizei1,Eigen::Dynamic,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator-=(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend SparseMatrix<std::common_type_t<T1,T2>,MatrixAdditionParams<Params1,Params2>::v,m_iIndexT1,m_jIndexT2>&
operator-=(SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);


//################## == operators
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend bool
operator==(Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
           const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

//##################Combined operators
template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Eigen::Vector<std::common_type_t<T1,T2>,sizei1> MulDiagonal(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A, const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Eigen::Vector<std::common_type_t<T1,T2>,sizei2> MulDiagonal(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A, const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Eigen::Vector<std::common_type_t<T1,T2>,sizei1> MulDiagonal(const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A, const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
friend Eigen::Vector<std::common_type_t<T1,T2>,Eigen::Dynamic> MulDiagonal(const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& A, const SparseMatrix<T2,Params2,m_iIndexT2,m_jIndexT2>& B);

template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1,
         typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2,
         typename T3, int sizei3, int sizej3, enums::MatrixProperties Params3, enums::IndexType m_iIndexT3, enums::IndexType m_jIndexT3>
friend Matrix<T3,sizei3,sizej3,Params3,m_iIndexT3,m_jIndexT3>&
PEAxB(Matrix<T3,sizei3,sizej3,Params3,m_iIndexT3,m_jIndexT3>& C,
      const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& A,
      const Matrix<T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>& B);

#endif


