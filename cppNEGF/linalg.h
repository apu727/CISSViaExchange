/*
 * This file contains the definitions for Matrix and sparseMatrix. It is sufficient to include this file to gain all functionality
 * Linalg.h should be included in files where the definitions are necessary
 * LinalgForwardDeclarations.h should be included in files where the declarations are necessary
 */

#ifndef LINALG_H
#define LINALG_H

#include "global.h"
#include "basismanager.h"
#include "complexmatrixbuffer.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include "linalgForwardDeclarations.h"
#define LINALG_DECLARE
#include "LinalgOperators.h"
#undef LINALG_DECLARE

//Wraps an Eigen::Matrix<T,size,size> and provides methods to compute the adjoint and basis changes automatically




inline void checkBasisChangeAssert(bool success, const indexBasisT& from, const indexBasisT& to)
{
    //Dont do the costly string manipulation if you dont have to
    if (success)
        return;
    std::string errorString = "Basis Change from " + from[0] + " to " + to[0] + " And " + from[0] + " to " + to[0]+ "  failed";
    releaseAssert(success,errorString);
}
inline void checkBasisChangeAssert(bool success, const std::string& from, const std::string& to){checkBasisChangeAssert(success,{from,to},{to,to});}





//Fixed size matrices also live in the buffer for now!
template <typename T, int sizei, int sizej, enums::MatrixProperties Params,
         enums::IndexType m_iIndexT, enums::IndexType m_jIndexT>
class Matrix
{
    //Convenience to extract the template parameters
public:
    typedef T static_T;
    static constexpr int static_sizei = sizei;
    static constexpr int static_sizej = sizej;
    static constexpr enums::MatrixProperties static_Params = Params;
    static constexpr enums::IndexType static_iIndexT = m_iIndexT;
    static constexpr enums::IndexType static_jIndexT = m_jIndexT;
    typedef Eigen::Matrix<T, sizei, sizej> MatType;

    typedef typename Eigen::NumTraits<T>::Real realT;
    typedef std::complex<realT> complexT;

    static constexpr bool isReal = std::is_same_v<realT,T>;
private:
    //Make every specialisation a friend
    template <typename, int, int, enums::MatrixProperties, enums::IndexType , enums::IndexType>
    friend class Matrix;
    template <typename, enums::MatrixProperties, enums::IndexType , enums::IndexType>
    friend class SparseMatrix;

    // friend BasisManager; //Such that it can access the buffer





    MatrixBufferObject<MatType> m_MatBuffer = MatrixBufferObject<MatType>::getEmpty();

    indexBasisT m_basis; // i index, j index basis. Allows for basisChange matrices to exist. Needed for some things to not be incredibly hacky
    enums::SpinSymmetry m_spinSym = enums::SpinSymmetry::NoSpin;

    long m_rows = sizei;
    long m_cols = sizej;

    Matrix(const std::string& basis,enums::SpinSymmetry sym) : Matrix({basis,basis},sym){}

    Matrix(const indexBasisT& basis,enums::SpinSymmetry sym)
    {
        //Allows setting spinSym and basis without giving it a buffer. Useful at times for checks
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        m_spinSym = sym;
        m_basis = basis;
    }
    Matrix(const MatrixBufferObject<MatType>& buf, const std::string& basis,enums::SpinSymmetry sym):Matrix(buf,{basis,basis},sym){}

    Matrix(const MatrixBufferObject<MatType>& buf, const indexBasisT& basis,enums::SpinSymmetry sym)
    {
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        m_spinSym = sym;
        m_basis = basis;
        m_MatBuffer = buf;
        m_rows = buf->rows();
        m_cols = buf->cols();
        **static_cast<const Matrix*>(this);
    }
    const MatType& operator *() const
    {
        return m_MatBuffer.get();
    }
public:
    static constexpr bool isSparse = false;
    Matrix(){};

    Matrix(long rows, long cols, const std::string& basis, enums::SpinSymmetry sym) :Matrix(rows,cols,{basis,basis},sym){}

    Matrix(long rows, long cols, const indexBasisT& basis, enums::SpinSymmetry sym)
    {
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        resize(rows,cols,basis,sym);
    }

    Matrix(long rows, std::string basis, enums::SpinSymmetry sym)
    {
        if constexpr(sizei == 1)
        {
            resize(1,rows,basis,sym);
        }
        else if constexpr (sizej == 1)
        {
            resize(rows,1,basis,sym);
        }
        else
            static_assert((sizei == 1 || sizej == 1) , "This method can only be called on vectors");


    }

    Matrix(MatType&& init, const std::string& basis, enums::SpinSymmetry sym) : Matrix(std::move(init),{basis,basis},sym){}

    Matrix(MatType&& init, const indexBasisT& basis, enums::SpinSymmetry sym) : m_MatBuffer()
    {
        m_spinSym = sym;
        m_basis = basis;
        m_rows = init.rows();
        m_cols = init.cols();
        *m_MatBuffer = std::move(init);
    }

    Matrix(const MatType& init, const std::string& basis, enums::SpinSymmetry sym) :  Matrix(init,{basis,basis},sym){}
    Matrix(const MatType& init, const indexBasisT& basis, enums::SpinSymmetry sym) : m_MatBuffer()
    {
        m_spinSym = sym;
        m_basis = basis;
        *m_MatBuffer = init;
        m_rows = init.rows();
        m_cols = init.cols();
    }

    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) = default;
    Matrix& operator= (Matrix&&) = default;
    Matrix& operator= (const Matrix&) = default;
    //Init functions
    //Better to move initialise since otherwise we may detach the buffer, copy just to set to zero
    void setZero() {*this = std::move(Zero(m_rows,m_cols,m_basis,m_spinSym));/*m_MatBuffer->setZero();*/ }
    void setRandom(){m_MatBuffer->setRandom();}
    void setZero(long rows, long cols, const std::string& basis, enums::SpinSymmetry sym) {setZero(rows,cols,{basis,basis},sym);}
    void setZero(long rows, long cols, const indexBasisT& basis, enums::SpinSymmetry sym) {*this = std::move(Zero(rows,cols,basis,sym));}

    void setIdentity(long rows, long cols, const std::string& basis, enums::SpinSymmetry sym) {setIdentity(rows,cols,{basis,basis},sym);}
    void setIdentity(long rows, long cols, const indexBasisT& basis, enums::SpinSymmetry sym) {*this = std::move(Identity(rows,cols,basis,sym));}

    void resize(long rows, long cols, const std::string& basis, enums::SpinSymmetry sym){resize(rows,cols,{basis,basis},sym);}
    void resize(long rows, long cols, const indexBasisT& basis, enums::SpinSymmetry sym)
    {
        if (m_MatBuffer.isEmpty())
            m_MatBuffer = MatrixBufferObject<MatType>();

        if constexpr (sizei != Eigen::Dynamic)
            releaseAssert(rows == sizei ,"row,cols constructor on fixed size matrix called with inconsistent sizei, sizei: " + std::to_string(sizei) + "rows: " + std::to_string(rows));
        if constexpr (sizej != Eigen::Dynamic)
            releaseAssert(cols == sizej ,"row,cols constructor on fixed size matrix called with inconsistent sizej, sizej: " + std::to_string(sizej) + "cols: " + std::to_string(cols));
        m_spinSym = sym;
        m_MatBuffer->resize(rows,cols);
        m_rows = rows;
        m_cols = cols;
        m_basis = basis;
    }

    //Operators
    //Does not support Eigen's lazy evaulation, yet
    friend std::ostream& operator<<(std::ostream& out, const Matrix& o)
    {
        if (o.m_MatBuffer.isEmpty())
        {
            return out << "Write empty Matrix";
        }
        return out << o.m_MatBuffer.get();
    }
    template<typename Derived>
    Matrix& operator=(const Eigen::MatrixBase<Derived>& x)
    { // is this a good idea? It allows assigning to this from eigen expressions which breaks encapsulation
        *m_MatBuffer = x;
        sizei = m_MatBuffer->rows();
        sizej = m_MatBuffer->cols();
    }

    //This is actually a good use for decltype(auto) because we want to pass the type of noalias exactly, https://stackoverflow.com/questions/24109737/what-are-some-uses-of-decltypeauto
    decltype(auto) noalias()
    { // is this a good idea? It allows assigning to this from eigen expressions which breaks encapsulation and we dont know what the user will use it for
        return m_MatBuffer->noalias();
    }

    const MatType& getEigenOptimisation() const
    { //High proability for misuse but needed to avoid lots of copies and stuff.
        return *m_MatBuffer;
    }
    MatType& getEigenOptimisation()
    { //High proability for misuse but needed to avoid lots of copies and stuff.
        return *m_MatBuffer;
    }

    friend class logger;
    friend int testBasisTransforms();
    friend int testBuiltinBasisTransforms();
    friend int testHardBasisTransforms();


#define LINALG_FRIEND
#include "LinalgOperators.h"
#undef LINALG_FRIEND



    //Maths Operators
    typename TraceReturn<Params,T>::t trace() const
    {
        // #warning add back
        static_assert(checkCompileTimeTrace<T,sizei,sizej,MatrixMultiplicationParams<Params,Params>::v,m_iIndexT,m_jIndexT>::v , "Matrix Trace Check Failed");
        checkRunTimeTrace(m_rows,m_cols,m_basis);

        if constexpr(std::is_same_v<typename TraceReturn<Params,T>::t,T>)
            return m_MatBuffer->trace();
        else
            return m_MatBuffer->trace().real();
    }

    //The trace is a scalar but the diagonal can transform weirdly?
    auto diagonal() const
    {
        //Although it may seem like this wont work, member functions of a templated class are only instantiated when called.
        //Therefore this code is not compiled until it is needed at which point it can fail
        //https://timsong-cpp.github.io/cppwp/n4618/temp.inst
        //https://stackoverflow.com/questions/26123254/member-function-instantiation
// #warning add back
        static_assert(checkCompileTimeTrace<T,sizei,sizej,MatrixMultiplicationParams<Params,Params>::v,m_iIndexT,m_jIndexT>::v , "Matrix Trace Check Failed");
        checkRunTimeTrace(m_rows,m_cols,m_basis);

        return m_MatBuffer->diagonal().eval();
    }
    //If the vector is an eigenvalue vector, The basis can be set here. This function probably only make sense for such a vector
    auto asDiagonal() const
    {
        return asDiagonal(m_basis);
    }
    auto asDiagonal(const std::string& desiredBasis) const {asDiagonal({desiredBasis,desiredBasis});}
    auto asDiagonal(const indexBasisT& desiredBasis) const
    {
        static_assert(sizei == 1 || sizej == 1, "CAN ONLY CALL asDiagonal ON A COMPILE TIME VECTOR");
        constexpr bool isReal = std::is_same_v<realT,T>;

        if constexpr (sizei == 1)
        {
            indexBasisT basis = m_basis;
            if constexpr(m_jIndexT == enums::IndexType::eigenValue)
                basis = desiredBasis;
            return Matrix<T,sizej,sizej,isReal ? enums::MatrixProperties::Hermitian : enums::MatrixProperties::None,m_jIndexT,m_jIndexT>(m_MatBuffer->asDiagonal(),basis,m_spinSym);
        }
        else if constexpr(sizej == 1)
        {
            indexBasisT basis = m_basis;
            if constexpr(m_iIndexT == enums::IndexType::eigenValue)
                basis = desiredBasis;
            return Matrix<T,sizei,sizei,isReal ? enums::MatrixProperties::Hermitian : enums::MatrixProperties::None,m_iIndexT,m_iIndexT>(m_MatBuffer->asDiagonal(),basis,m_spinSym);
        }
        else
            static_assert(dependent_false<decltype(*this)>::value, "UNREACHABLE");


    }

    //Helper typedef for Eigenvalues
    typedef Matrix<typename EigReturn<Matrix>::Val,sizei,1,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::invalid> EigenValueType;
    typedef Matrix<typename EigReturn<Matrix>::Vec,sizei,sizej, EigReturn<Matrix>::VecParams,indexContractionPair<m_jIndexT>::v,enums::IndexType::eigenValue> EigenVectorType;
    //Check if non-square
    bool getEigenValues(EigenValueType& eValues) const
    {
        EigReturn<Matrix>::check();
        if constexpr (sizei != Eigen::Dynamic && sizej != Eigen::Dynamic)
            static_assert(sizei == sizej,"cannot get EigenValues of nonSquareMatrix");
        releaseAssert(m_rows == m_cols,"cannot get EigenValues of nonSquareMatrix");
        releaseAssert(m_basis[0] == m_basis[1],"cannot get EigenValues of a matrix with indexes in different bases");


        if constexpr(Params == enums::MatrixProperties::Hermitian)
        {
            //Know we are in a direct dual basis or something similar.
            Eigen::SelfAdjointEigenSolver<MatType> es(*m_MatBuffer,Eigen::EigenvaluesOnly);
            eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues(),"",m_spinSym); // EigenValues dont have a basis
            return true;

        }
        else if constexpr(Params == enums::MatrixProperties::selfAdjoint)
        {
            if (BasisManager::getInstance().isOrthogonal(m_basis[0]))
            {
                Eigen::SelfAdjointEigenSolver<MatType> es(*m_MatBuffer,Eigen::EigenvaluesOnly);
                eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues(),"",m_spinSym); // EigenValues dont have a basis
                return true;
            }
            else
            {
                Matrix ThisInOrthogonalBasis;
                bool success = toOrthogonalBasis(ThisInOrthogonalBasis);
                if (!success)
                    return false;
                success = ThisInOrthogonalBasis.getEigenValues(eValues);
                return success;
            }
        }
        Eigen::ComplexEigenSolver<MatType> es(*m_MatBuffer,Eigen::EigenvaluesOnly);
        if constexpr (EigReturn<Matrix>::isSelfAdjoint || EigReturn<Matrix>::isHermitian)
            eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues().real(),"",m_spinSym); // EigenValues dont have a basis
        else
            eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues(),"",m_spinSym); // EigenValues dont have a basis
        return true;

    }

    [[nodiscard]] bool getEigenValuesAndVectors(EigenValueType& eValues,
                                  EigenVectorType& eVecs, bool diagonalConditioner = false) const
    {
        EigReturn<Matrix>::check();
        if constexpr (sizei != Eigen::Dynamic && sizej != Eigen::Dynamic)
            static_assert(sizei == sizej,"cannot get EigenValues of nonSquareMatrix");
        releaseAssert(m_rows == m_cols,"cannot get EigenValues of nonSquareMatrix");
        releaseAssert(m_basis[0] == m_basis[1],"cannot get EigenValues of a matrix with indexes in different bases");

        if constexpr (isReal)
            if (diagonalConditioner) logger().log("Diagonal conditioner not usable with real matrices");

        if (m_spinSym == enums::SpinSymmetry::RHF && !BasisManager::hasSpinBlockTag(m_basis[0]))
        {
            //Fail for compileTimeSizedMatrices
            //Cant be static assert since it depends on the runtime spin type
            releaseAssert(m_rows %2 == 0 &&  m_cols %2 == 0,"RHF matrix with odd rows/cols");

            //This is needed because toBasis wont let Hermitian stay Hermitian even tho in this case it is
            typedef typename changeMatrixArgs<Matrix,complexT,sizei,sizej,enums::MatrixProperties::None>::Dense NoParamTypedef;
            typename SpinBlocking<NoParamTypedef,NoParamTypedef>::DenseResult mat;

            toBasisC(mat,m_basis[0] + BasisManager::Alpha_Block);


            typename changeMatrixArgs<decltype(mat),T,mat.static_sizei,mat.static_sizej,Params>::Dense::EigenValueType tempEVals;
            typename changeMatrixArgs<decltype(mat),T,mat.static_sizei,mat.static_sizej,Params>::Dense::EigenVectorType tempeVecs;

            bool ret;
            if constexpr(isReal)
                ret = typename changeMatrixArgs<decltype(mat),T,mat.static_sizei,mat.static_sizej,Params>::Dense(mat.m_MatBuffer.get().real(),mat.getBasis(),m_spinSym)
                          .getEigenValuesAndVectors(tempEVals,tempeVecs,diagonalConditioner);
            else
                ret = typename changeMatrixParameters<decltype(mat),Params>::Dense(mat.m_MatBuffer.get(),mat.getBasis(),m_spinSym)
                          .getEigenValuesAndVectors(tempEVals,tempeVecs,diagonalConditioner);
            if (!ret) return false;

            //Eigenvalues kinda break in basis transforms because they do have spin properties but not basis properties. Would need to implement this properly. For now:
            auto transformEValAlpha = BasisManager::getInstance().getTransform(m_basis[0] + BasisManager::Alpha_Block,enums::IndexType::direct,m_basis[0],enums::IndexType::direct);
            if (!transformEValAlpha) return false;
            auto transformEValBeta = BasisManager::getInstance().getTransform(m_basis[0] + BasisManager::Beta_Block,enums::IndexType::direct,m_basis[0],enums::IndexType::direct);
            if (!transformEValBeta) return false;

            auto transformEVecAlpha = BasisManager::getInstance().getTransform(m_basis[0] + BasisManager::Alpha_Block,eVecs.static_iIndexT,m_basis[0],eVecs.static_iIndexT);
            if (!transformEVecAlpha) return false;
            auto transformEVecBeta = BasisManager::getInstance().getTransform(m_basis[0] + BasisManager::Beta_Block,eVecs.static_iIndexT,m_basis[0],eVecs.static_iIndexT);
            if (!transformEVecBeta) return false;

            eValues = remove_cvref_t<decltype(eValues)>(tempEVals.m_MatBuffer->transpose() * transformEValAlpha->real(),"",m_spinSym);
            eValues.m_MatBuffer->noalias() += tempEVals.m_MatBuffer->transpose() * transformEValBeta->real();
            //FIXME creates a temporary
            eVecs = remove_cvref_t<decltype(eVecs)>(transformEVecAlpha->real().transpose() * tempeVecs.m_MatBuffer.get() * transformEValAlpha->real(),m_basis,m_spinSym); // Basis is wrong here but eigenvalue types dont get transformed so its fine.
            eVecs.m_MatBuffer->noalias() += transformEVecBeta->real().transpose() * tempeVecs.m_MatBuffer.get() * transformEValBeta->real();
            return true;
        }

        if constexpr(Params == enums::MatrixProperties::Hermitian)
        {

            Eigen::SelfAdjointEigenSolver<MatType> es(*m_MatBuffer,Eigen::ComputeEigenvectors);
            eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues(),"",m_spinSym); // EigenValues dont have a basis
            eVecs = remove_cvref_t<decltype(eVecs)>(es.eigenvectors(),m_basis,m_spinSym); // The vectors themselves have a basis
            return true;
        }
        else
        {
            if constexpr(Params == enums::MatrixProperties::selfAdjoint)
            {
                if (BasisManager::getInstance().isOrthogonal(m_basis[0]))
                {
                    Eigen::SelfAdjointEigenSolver<MatType> es(*m_MatBuffer,Eigen::ComputeEigenvectors);
                    eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues(),"",m_spinSym); // EigenValues dont have a basis
                    eVecs = remove_cvref_t<decltype(eVecs)>(es.eigenvectors(),m_basis,m_spinSym); // The vectors themselves have a basis
                    return true;
                }
                else
                {
                    Matrix ThisInOrthogonalBasis;
                    bool success = toOrthogonalBasis(ThisInOrthogonalBasis);
                    if (!success) return false;
                    success = ThisInOrthogonalBasis.getEigenValuesAndVectors(eValues,eVecs,diagonalConditioner);
                    if (!success) return false;
                    eVecs.toBasisC(eVecs,m_basis);
                    return true;
                }
            }
            Eigen::ComplexEigenSolver<MatType> es;
            if constexpr (isReal)
            {
                es.compute(*m_MatBuffer,Eigen::ComputeEigenvectors);
            }
            else
            {
                if (diagonalConditioner)
                    es.compute(*m_MatBuffer - iu*MatType::Identity(m_rows,m_cols),Eigen::ComputeEigenvectors);
                else
                    es.compute(*m_MatBuffer,Eigen::ComputeEigenvectors);
            }

            if constexpr (EigReturn<Matrix>::isSelfAdjoint || EigReturn<Matrix>::isHermitian)
            {
                if constexpr(isReal)
                {
                    eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues().real(),"",m_spinSym); // EigenValues dont have a basis
                }
                else
                {
                    if (diagonalConditioner)
                        eValues = remove_cvref_t<decltype(eValues)>((es.eigenvalues().array() + iu).real(),"",m_spinSym); // EigenValues dont have a basis
                    else
                        eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues().real(),"",m_spinSym); // EigenValues dont have a basis
                }
            }
            else
            {
                if constexpr (isReal)
                {
                    eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues(),"",m_spinSym); // EigenValues dont have a basis
                }
                else
                {
                    if (diagonalConditioner)
                        eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues().array() + iu,"",m_spinSym); // EigenValues dont have a basis
                    else
                        eValues = remove_cvref_t<decltype(eValues)>(es.eigenvalues(),"",m_spinSym); // EigenValues dont have a basis
                }
            }
            eVecs = remove_cvref_t<decltype(eVecs)>(es.eigenvectors(),m_basis,m_spinSym); // The vectors themselves have a basis
            if (!diagonalConditioner)
            {
                const realT machinePrecision = 1e-14;
                bool foundSmallEigenValue = false;
                typename EigReturn<Matrix>::Val smallValue;
                for (const auto& e : eValues.m_MatBuffer.get())
                {
                    if (abs(e) < machinePrecision)
                    {
                        foundSmallEigenValue = true;
                        smallValue = e;
                        break;
                    }
                }
                if (foundSmallEigenValue)
                    logger().log("Small EigenValue found, Eigenvectors may not be stable, suggest setting Diagonal Conditioner to true",smallValue);
            }
            return true;

        }

    }

    //Basis transformations
    //Returns false if not successful
    template <typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    [[nodiscard]] bool toBasis(Matrix<T2,sizei2,sizej2, Params2,m_iIndexT2,m_jIndexT2>& dest, const std::string& desiredBasis,bool silenceSpinBlockWarning = false) const
    {return toBasis(dest,{desiredBasis,desiredBasis},silenceSpinBlockWarning);}

    template <typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    [[nodiscard]] bool toBasis(Matrix<T2,sizei2,sizej2, Params2,m_iIndexT2,m_jIndexT2>& dest, const indexBasisT& desiredBasis,bool silenceSpinBlockWarning = false) const
    {

        typedef remove_cvref_t<decltype(dest)> destType;
        typedef remove_cvref_t<decltype(*this)> myType;

        static_assert(checkBasisReturn<T,sizei,sizej,Params,m_iIndexT,m_jIndexT,T2,sizei2,sizej2, Params2,m_iIndexT2,m_jIndexT2>::v , "Basis change compile time check failed");

        if constexpr(std::is_constructible_v<myType,destType>/* Evaluates to true if there is a cast from myType to destType*/)
        {
            if (m_basis == desiredBasis)
            {
                logger().log("Uneccessary Basis conversion");
                dest = static_cast<destType>(*this);
                return true;
            }
        }
        //If one is dynamic then they could be the same size and Eigen wont complain. However MatrixBufferObject will.
        // error: no match for ‘operator=’ (operand types are ‘MatrixBufferObject<Eigen::Matrix<std::complex<double>, -1, 1> >’ and ‘MatrixBufferObject<Eigen::Matrix<std::complex<double>, -1, -1> >’)
        constexpr bool sizeiIsSame = sizei == sizei2 ;//|| ((sizei == Eigen::Dynamic) != (sizei2 == Eigen::Dynamic));
        constexpr bool sizejIsSame = sizej == sizej2 ;//|| ((sizej == Eigen::Dynamic) != (sizej2 == Eigen::Dynamic));

        //If one of these is false then we have to assume the transformation is not the identity. If it is then we fail at runtime

        //Sort out the spin tags
        auto& BM = BasisManager::getInstance();
        bool validTags = false;
        auto desiredBasisiTags = BM.checkForTags(desiredBasis[0],&validTags);
        if (!validTags)
        {
            logger().log("Invalid Tags on basis", desiredBasis[0]);
            return false;
        }
        auto desiredBasisjTags = BM.checkForTags(desiredBasis[1],&validTags);
        if (!validTags)
        {
            logger().log("Invalid Tags on basis", desiredBasis[1]);
            return false;
        }

        auto myBasisiTags = BM.checkForTags(m_basis[0],&validTags);
        if (!validTags)
        {
            logger().log("Invalid Tags on basis", m_basis[0]);
            return false;
        }
        auto myBasisjTags = BM.checkForTags(m_basis[1],&validTags);
        if (!validTags)
        {
            logger().log("Invalid Tags on basis", m_basis[1]);
            return false;
        }

        bool IHaveA_i = BasisManager::contains(myBasisiTags,BasisManager::possibleTags::ABlock);
        bool IHaveB_i = BasisManager::contains(myBasisiTags,BasisManager::possibleTags::BBlock);

        bool IHaveA_j = BasisManager::contains(myBasisjTags,BasisManager::possibleTags::ABlock);
        bool IHaveB_j = BasisManager::contains(myBasisjTags,BasisManager::possibleTags::BBlock);

        bool desiredHasA_i = BasisManager::contains(desiredBasisiTags,BasisManager::possibleTags::ABlock);
        bool desiredHasB_i = BasisManager::contains(desiredBasisiTags,BasisManager::possibleTags::BBlock);

        bool desiredHasA_j = BasisManager::contains(desiredBasisjTags,BasisManager::possibleTags::ABlock);
        bool desiredHasB_j = BasisManager::contains(desiredBasisjTags,BasisManager::possibleTags::BBlock);

        if (IHaveA_i || IHaveA_j || IHaveB_i || IHaveB_j || desiredHasA_i || desiredHasA_j || desiredHasB_i || desiredHasB_j)
        {//Worry about spin
            //There are lots of headaches with using compile sized matrices here mainly due to getting their dimensions correct
            //at compile time but not knowing the dimensions at compile time as it depends on the spin transformation performed
            //for this reason we convert all compile time matrices to dynamic and call ourselves again, Nx1 vectors are fine
            // if constexpr(sizei == sizej && sizei != Eigen::Dynamic)
            // {
            //     typename changeMatrixArgs<Matrix,T,Eigen::Dynamic,Eigen::Dynamic>::Dense temp(m_MatBuffer.get(),m_basis,m_spinSym);
            //     return temp.toBasis(dest,desiredBasis);
            // }
            // else
            // {
                if (m_spinSym != enums::SpinSymmetry::RHF && m_spinSym != enums::SpinSymmetry::NoSpinSym)
                    releaseAssert(false,"Requested a spin transformation for a matrix without spin");

                enum spinType
                {
                    Alpha,
                    Beta,
                    Both,
                };

                spinType myLeftIndex = Both;
                spinType myRightIndex = Both;
                spinType desiredLeftIndex = Both;
                spinType desiredRightIndex = Both;

                if (IHaveA_i)
                    myLeftIndex = Alpha;

                if (IHaveB_i)
                    myLeftIndex = Beta;

                if (IHaveA_j)
                    myRightIndex = Alpha;

                if (IHaveB_j)
                    myRightIndex = Beta;

                if (desiredHasA_i)
                    desiredLeftIndex = Alpha;

                if (desiredHasB_i)
                    desiredLeftIndex = Beta;

                if (desiredHasA_j)
                    desiredRightIndex = Alpha;

                if (desiredHasB_j)
                    desiredRightIndex = Beta;

                BasisManager::possibleTags tagFrom_i = BasisManager::possibleTags::NullBasis;
                BasisManager::possibleTags tagFrom_j = BasisManager::possibleTags::NullBasis;
                BasisManager::possibleTags tagTo_i = BasisManager::possibleTags::NullBasis;
                BasisManager::possibleTags tagTo_j = BasisManager::possibleTags::NullBasis;

                std::string noSpinTagFrom_i = BasisManager::removeSpinTags(m_basis[0],&tagFrom_i);
                std::string noSpinTagFrom_j = BasisManager::removeSpinTags(m_basis[1],&tagFrom_j);

                std::string noSpinTagto_i = BasisManager::removeSpinTags(desiredBasis[0],&tagTo_i);
                std::string noSpinTagto_j = BasisManager::removeSpinTags(desiredBasis[1],&tagTo_j);
                //Not strictly necessary as CHECK does this warning too
                if (!silenceSpinBlockWarning && m_spinSym != enums::SpinSymmetry::RHF && (myLeftIndex != Both || myRightIndex != Both))
                    logger().log("Requesting a Non RHF matrix come out of a spin block. This is probably not what you wanted");

                //If noSpinTagFrom and nospinTagto are in different sized bases to AND none of the matrices are dynamic then this will fail at runtime
                //as we have no way of knowing the size of *this before/after blocking
                //Further problems dest = 1x1 matrix, src = 2x2 matrix
                //transformi is never the identity in a well defined situation. e.g go to AA Block of 2x2 matrix.
                //But we cannot know this at compile time since the well definedness of the situation is determined at runtime
                //Therefore to avoid all these problems and more! We always use dynamic matrices here.

                //Pretend we always unblock and then block. If either one isnt actually a block then the check succeeds.
                typedef typename SpinUnBlocking<Matrix,destType>::DenseResult IntermediateType;
                IntermediateType pseudoIntermediate1({noSpinTagFrom_i,noSpinTagFrom_j},m_spinSym);
                SpinUnBlocking<Matrix,IntermediateType>::check(*this,pseudoIntermediate1,{tagFrom_i,tagFrom_j},silenceSpinBlockWarning);
                IntermediateType pseudoIntermediate2({noSpinTagto_i,noSpinTagto_j},m_spinSym);
                SpinBlocking<IntermediateType,destType>::check(pseudoIntermediate2,destType(desiredBasis,m_spinSym),{tagTo_i,tagTo_j});


                std::string leftFrom = noSpinTagFrom_i;
                if (myLeftIndex == Alpha)
                    leftFrom += BasisManager::Alpha_Block;
                if (myLeftIndex == Beta)
                    leftFrom += BasisManager::Beta_Block;

                std::string rightFrom = noSpinTagFrom_j;
                if (myRightIndex == Alpha)
                    rightFrom += BasisManager::Alpha_Block;
                if (myRightIndex == Beta)
                    rightFrom += BasisManager::Beta_Block;


                std::string leftTo = noSpinTagto_i;
                if (desiredLeftIndex == Alpha)
                    leftTo += BasisManager::Alpha_Block;
                if (desiredLeftIndex == Beta)
                    leftTo += BasisManager::Beta_Block;

                std::string rightTo = noSpinTagto_j;
                if (desiredRightIndex == Alpha)
                    rightTo += BasisManager::Alpha_Block;
                if (desiredRightIndex == Beta)
                    rightTo += BasisManager::Beta_Block;


                auto transformi = BM.getTransform(leftFrom,m_iIndexT,leftTo,m_iIndexT2);
                if (!transformi)
                    return false;
                auto transformj = BM.getTransform(rightFrom,m_jIndexT,rightTo,m_jIndexT2);
                if (!transformi)
                    return false;

                MatrixBufferObject<Eigen::Matrix<T2,sizei2,sizej>> temp;
                MatrixBufferObject<typename destType::MatType> tempDest; // need to create a tempDest because dest may alias *this
                if constexpr(sizeiIsSame)
                {
                    if (transformi.isIdentity || sizei == 1)
                        temp.get() =  m_MatBuffer.get();
                    else
                        temp.get() = transformi->transpose() * m_MatBuffer.get();
                }
                else
                {
                    if (transformi.isIdentity)
                    {
                        std::string errorMessage = "Transformation from '" + leftFrom + "' to '" + leftTo + "' is the identity in i. But, sizeiIn is: " + std::to_string(sizei) + "sizei out: " + std::to_string(sizei2);
                        logger().log(errorMessage);
                        return false;
                    }
                    temp.get() = transformi->transpose() * m_MatBuffer.get();
                }
                if constexpr(sizejIsSame)
                {
                    if (transformj.isIdentity || sizej == 1)
                        tempDest = temp;
                    else
                        tempDest->noalias() = temp.get()* *transformj;
                }
                else
                {
                    if (transformj.isIdentity)
                    {
                        std::string errorMessage = "Transformation from '" + rightFrom + "' to '" + rightTo + "' is the identity in j. But, sizejIn is: " + std::to_string(sizej) + "sizej out: " + std::to_string(sizej2);
                        logger().log(errorMessage);
                        return false;
                    }
                    tempDest->noalias() = temp.get()* *transformj;
                }
                if (m_spinSym == enums::SpinSymmetry::RHF && (myLeftIndex != Both || myRightIndex != Both))
                {
                    //Also have to pretend we have a beta_beta block
                    leftFrom = noSpinTagFrom_i;
                    if (myLeftIndex == Beta)
                        leftFrom += BasisManager::Alpha_Block;
                    if (myLeftIndex == Alpha)
                        leftFrom += BasisManager::Beta_Block;

                    rightFrom = noSpinTagFrom_j;
                    if (myRightIndex == Beta)
                        rightFrom += BasisManager::Alpha_Block;
                    if (myRightIndex == Alpha)
                        rightFrom += BasisManager::Beta_Block;

                    auto transformi = BM.getTransform(leftFrom,m_iIndexT,leftTo,m_iIndexT2);
                    if (!transformi)
                        return false;
                    if (myLeftIndex == Both)
                        transformi.isIdentity = true;

                    auto transformj = BM.getTransform(rightFrom,m_jIndexT,rightTo,m_jIndexT2);
                    if (!transformj)
                        return false;
                    if (myRightIndex == Both)
                        transformj.isIdentity = true;

                    if constexpr(sizeiIsSame)
                    {
                        if (transformi.isIdentity || sizei == 1)
                            temp.get() =  m_MatBuffer.get();
                        else
                            temp.get() = transformi->transpose() * m_MatBuffer.get();
                    }
                    else
                    {
                        if (transformi.isIdentity)
                        {
                            std::string errorMessage = "Transformation from '" + leftFrom + "' to '" + leftTo + "' is the identity in i. But, sizeiIn is: " + std::to_string(sizei) + "sizei out: " + std::to_string(sizei2);
                            logger().log(errorMessage);
                            return false;
                        }
                        temp.get() = transformi->transpose() * m_MatBuffer.get();
                    }
                    if constexpr(sizejIsSame)
                    {
                        if (transformj.isIdentity || sizej == 1)
                            tempDest->noalias() += temp.get();
                        else
                            tempDest->noalias() += temp.get()* *transformj;
                    }
                    else
                    {
                        if (transformj.isIdentity)
                        {
                            std::string errorMessage = "Transformation from '" + rightFrom + "' to '" + rightTo + "' is the identity in j. But, sizejIn is: " + std::to_string(sizej) + "sizej out: " + std::to_string(sizej2);
                            logger().log(errorMessage);
                            return false;
                        }
                        tempDest->noalias() += temp.get()* *transformj;
                    }
                }
                dest = destType(tempDest,desiredBasis,m_spinSym);
                return true;
            // }
        }
        else
        {
            //No spin nonsense
            //For this reason the returned matrix should be used as v_mu = v_i Mat^i_mu
            auto transformi = BM.getTransform(m_basis[0],m_iIndexT,desiredBasis[0],m_iIndexT2);
            if (!transformi)
                return false;
            auto transformj = BM.getTransform(m_basis[1],m_jIndexT,desiredBasis[1],m_jIndexT2);
            if (!transformj)
                return false;

            MatrixBufferObject<Eigen::Matrix<T2,sizei2,sizej>> temp;
            if constexpr(sizeiIsSame)
            {
                if (transformi.isIdentity || sizei == 1)
                    temp.get() =  m_MatBuffer.get();
                else
                    temp.get() = transformi->transpose() * m_MatBuffer.get();
            }
            else
            {
                if (transformi.isIdentity)
                {
                    std::string errorMessage = "Transformation from '" + m_basis[0] + "' to '" + desiredBasis[0] + "' is the identity in i. But, sizeiIn is: " + std::to_string(sizei) + "sizei out: " + std::to_string(sizei2);
                    logger().log(errorMessage);
                    return false;
                }
                temp.get() = transformi->transpose() * m_MatBuffer.get();
            }
            if constexpr(sizejIsSame)
            {
                if (transformj.isIdentity || sizej == 1)
                    dest = destType(temp,desiredBasis,m_spinSym);
                else
                    dest = destType(temp.get()* *transformj,desiredBasis,m_spinSym);
            }
            else
            {
                if (transformj.isIdentity)
                {
                    std::string errorMessage = "Transformation from '" + m_basis[1] + "' to '" + desiredBasis[1] + "' is the identity in j. But, sizejIn is:" + std::to_string(sizej) + "sizej out:" + std::to_string(sizej2);
                    logger().log(errorMessage);
                    return false;
                }
                dest = destType(temp.get()* *transformj,desiredBasis,m_spinSym);
            }
            return true;
        }

    }

    template <typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    void toBasisC(Matrix<T2,sizei2,sizej2, Params2,m_iIndexT2,m_jIndexT2>& dest, const std::string& desiredBasis) const {toBasisC(dest,{desiredBasis,desiredBasis});}

    //Same as toBasis but calls the check function automatically. Convenience
    template <typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    void toBasisC(Matrix<T2,sizei2,sizej2, Params2,m_iIndexT2,m_jIndexT2>& dest, const indexBasisT& desiredBasis) const
    {
        bool success = toBasis(dest,desiredBasis);
        checkBasisChangeAssert(success,m_basis,desiredBasis);
    }



    //Error if not marked as self adjoint | enums::MatrixProperties::Hermitian

    [[nodiscard]] bool toOrthogonalBasis(Matrix& dest) const
    {
        if (BasisManager::getInstance().isOrthogonal(m_basis[0]) && BasisManager::getInstance().isOrthogonal(m_basis[1]))
            dest = *this;
        else
        {
            indexBasisT newBasis = m_basis;
            if (!BasisManager::getInstance().isOrthogonal(m_basis[0]))
                newBasis[0] = m_basis[0] + BasisManager::LowdinOrthogonaliseBasis;
            if (!BasisManager::getInstance().isOrthogonal(m_basis[1]))
                newBasis[1] = m_basis[1] + BasisManager::LowdinOrthogonaliseBasis;

            Matrix<T,sizei,sizej,enums::MatrixProperties::None,m_iIndexT,m_jIndexT> temp;
            bool success = toBasis(temp,newBasis);
            if (!success)
                return success;
            dest = static_cast<Matrix>(temp);
        }
        return true;

    }

    [[nodiscard]] bool toHermitianBasis(typename HermReturn<T,sizei,sizej,Params,m_iIndexT,m_jIndexT>::Mat& dest) const
    {
        //A very weird object if the bases arent the same but possible?
        releaseAssert(m_basis[0] == m_basis[1],"ToHermitian Basis with differing index basis");

        HermReturn<T,sizei,sizej,Params,m_iIndexT,m_jIndexT>::check();
        if constexpr (m_iIndexT == enums::IndexType::dual && m_jIndexT == enums::IndexType::direct && Params == enums::MatrixProperties::Hermitian)
        {
            logger().log("Already in Hermitian basis, useless transform");
            dest = *this;
            return true;
        }
        else
        {

            if (BasisManager::contains(BasisManager::checkForTags(m_basis[0]),BasisManager::possibleTags::LowdinBasis))
                dest = static_cast<typename HermReturn<T,sizei,sizej,Params,m_iIndexT,m_jIndexT>::Mat>(*this);
            else
            {
                Matrix<T,sizei,sizej,enums::MatrixProperties::None,m_iIndexT,m_jIndexT> temp;
                bool success = toBasis(temp,m_basis[0] + BasisManager::LowdinOrthogonaliseBasis);
                if (!success)
                    return success;
                dest = static_cast<typename HermReturn<T,sizei,sizej,Params,m_iIndexT,m_jIndexT>::Mat>(temp);
            }
            return true;
        }
    }

    void makeSelfAdjoint(bool multiplyBy2 = false)
    {
        releaseAssert(m_basis[0] == m_basis[1],"makeSelfAdjoint Basis with differing index basis");
        if (BasisManager::getInstance().isOrthogonal(m_basis[0]))
        {
            for (long i = 0; i < m_rows; i++)
            {
                for (long j = 0; j < i; j++)
                {
                    if constexpr(isReal)
                        (*this)(j,i) = (*this)(i,j);
                    else
                        (*this)(j,i) = std::conj((*this)(i,j));
                }
                if constexpr(!isReal)
                {
                    (*this)(i,i) = (*this)(i,i).real();
                }
            }
            if (multiplyBy2)
                *this *= 2;
        }
        else
        {
            *this += this->adjoint();
            if (!multiplyBy2)
                *this /=2;
        }
    }

    auto adjoint() const
    { // is the adjoint of a Unitary, Unitary - Probably
        if constexpr (Params == enums::MatrixProperties::selfAdjoint && indexContractionPair<m_jIndexT>::v == m_iIndexT && sizei == sizej)
        {
            if constexpr(sizei == -1 && sizej == -1)
                releaseAssert(m_rows==m_cols,"Self Adjoint but not square matrix doesnt make sense");
            return *this;
        }
        else
        {
            if constexpr(Params == enums::MatrixProperties::Hermitian)
                if (BasisManager::getInstance().isOrthogonal(m_basis[0]) && m_basis[0] == m_basis[1])
                    return Matrix<complexT,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v>(m_MatBuffer.get().template cast<complexT>(),m_basis,m_spinSym);

            //The following code lowers the indexes, takes the conjugate transpose, This can be skipped if it is orthogonal
            if (BasisManager::getInstance().isOrthogonal(m_basis[0]) && BasisManager::getInstance().isOrthogonal(m_basis[1]))
                return Matrix<complexT,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v>(m_MatBuffer->adjoint(),{m_basis[1],m_basis[0]},m_spinSym);
            Matrix<complexT,sizei,sizej,enums::MatrixProperties::None,indexConjugatePair<indexContractionPair<m_iIndexT>::v>::v,indexConjugatePair<indexContractionPair<m_jIndexT>::v>::v> partialAdjoint;
            toBasisC(partialAdjoint,m_basis);
            Matrix<complexT,sizej,sizei,enums::MatrixProperties::None,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v> CanonicalAdjoint(partialAdjoint.m_MatBuffer->adjoint(),{m_basis[1],m_basis[0]},m_spinSym);

            return static_cast<Matrix<complexT,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v>>(CanonicalAdjoint);
        }
    }

    auto conjugateTranspose() const
    {
        return Matrix<T,sizej,sizei,Params,indexConjugatePair<m_jIndexT>::v,indexConjugatePair<m_iIndexT>::v>(m_MatBuffer->adjoint(),m_basis,m_spinSym);
    }

    Matrix conjugate() const
    {
        return Matrix<T,sizei,sizej,Params,indexConjugatePair<m_iIndexT>::v,indexConjugatePair<m_jIndexT>::v>(m_MatBuffer->conjugate(),m_basis,m_spinSym);
    }

    realT norm()//TODO const
    {
        static_assert(checkCompileTimeMatrixMultiplication<complexT,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v
                                                           ,T,sizei,sizej,Params,m_iIndexT,m_jIndexT>::v , "Matrix Multiplication Check Failed ");
        static_assert(checkCompileTimeTrace<complexT,sizej,sizej,MatrixMultiplicationParams<Params,Params>::v,indexContractionPair<m_jIndexT>::v,m_jIndexT>::v , "Matrix Trace Check Failed");
        checkRunTimeMatrixMultiplication(*this,*this);
        checkRunTimeTrace(m_rows,m_cols,m_basis);
        if constexpr (Params == enums::MatrixProperties::selfAdjoint)
            return m_MatBuffer->norm();
        else
        {
            if (!BasisManager::getInstance().isOrthogonal(m_basis[0]) || !BasisManager::getInstance().isOrthogonal(m_basis[1]))
                logger().log("Norm of Non self adjoint matrix can be slow in non orthogonal basis");
            return std::sqrt(std::real(this->cdot(*this)));
        }
    }
    realT squaredNorm()
    {
        static_assert(checkCompileTimeMatrixMultiplication<complexT,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v
                                                           ,T,sizei,sizej,Params,m_iIndexT,m_jIndexT>::v , "Matrix Multiplication Check Failed ");
        static_assert(checkCompileTimeTrace<complexT,sizej,sizej,MatrixMultiplicationParams<Params,Params>::v,indexContractionPair<m_jIndexT>::v,m_jIndexT>::v , "Matrix Trace Check Failed");
        checkRunTimeMatrixMultiplication(*this,*this);
        checkRunTimeTrace(m_rows,m_cols,m_basis);

        if constexpr (Params == enums::MatrixProperties::selfAdjoint)
            return m_MatBuffer->squaredNorm();
        else
        {
            if (!BasisManager::getInstance().isOrthogonal(m_basis[0]) || !BasisManager::getInstance().isOrthogonal(m_basis[1]))
                logger().log("Norm of Non self adjoint matrix can be slow in non orthogonal basis");
            return std::real(this->cdot(*this));
        }

    }

    template <typename T2, int sizei2, int sizej2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    auto cdot(const Matrix<T2,sizei2,sizej2, Params2,m_iIndexT2,m_jIndexT2>& other) const
    {
        if constexpr(sizei2 == 2 && sizej2 == 2)
        {//Special case the Spin matrix multiply
            releaseAssert(false,"Spin Matrix multiply not implemented yet");
            return realT();
        }
        else if constexpr(sizei == 2 && sizej == 2)
        {//Special case the Spin matrix multiply
            releaseAssert(false,"Spin Matrix multiply not implemented yet");
            return realT();
        }
        else
        {
            // The adjoint type is static_cast<Matrix<complexT,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v>>
            static_assert(checkCompileTimeMatrixMultiplication<complexT,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v
                                                               ,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed ");
            static_assert(checkCompileTimeTrace<complexT,sizej,sizej2,MatrixMultiplicationParams<Params,Params2>::v,indexContractionPair<m_jIndexT>::v,m_jIndexT2>::v , "Matrix Trace Check Failed");

            checkRunTimeMatrixMultiplication(*this,other);
            checkRunTimeTrace(m_rows,other.m_cols,{m_basis[0],other.m_basis[1]});
            if constexpr (Params == enums::MatrixProperties::selfAdjoint)
                return (m_MatBuffer.get() * other.m_MatBuffer.get()).trace();
            else
            {
                if (BasisManager::getInstance().isOrthogonal(m_basis[0]) && BasisManager::getInstance().isOrthogonal(m_basis[1]))
                    return complexT((m_MatBuffer->adjoint() * other.m_MatBuffer.get()).trace());
                return (adjoint().m_MatBuffer.get() * other.m_MatBuffer.get()).trace();
            }
        }
    }

    Matrix<T,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v> inverse() const
    {
        if constexpr(Params == enums::MatrixProperties::Unitary)
        {
            return Matrix<T,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v>(m_MatBuffer->adjoint(),m_basis,m_spinSym);
        }
        else if (Params == enums::MatrixProperties::NormPreserving)
        {
            //Since this must be the same as inverse.The inverse of a real matrix is real and so the adjoint must also be
            if constexpr(std::is_same_v<realT,T>)
                return Matrix<T,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v>(adjoint().m_MatBuffer.get().real(),m_basis,m_spinSym);
            else
                return adjoint();
        }
        else
        {
            return Matrix<T,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v>(m_MatBuffer->inverse(),{m_basis[1],m_basis[0]},m_spinSym);
        }
    }

    Matrix<T,sizej,sizei,Params,m_jIndexT,m_iIndexT> transpose()
    {
        return Matrix<T,sizej,sizei,Params,m_jIndexT,m_iIndexT>(m_MatBuffer->transpose(),{m_basis[1],m_basis[0]},m_spinSym);
    }
    long rows()const {return m_rows;}
    long cols()const {return m_cols;}

    //This is technically an escape hatch by which the matrix can be accessed directly. Nevertheless you really have to try and it would be inconvenient to patch it
    //Further if we allow the .coeffRef(i,j) operator then you can do whatever anywa so this is not a conceptual flaw

    decltype(auto) col(long i) const {return m_MatBuffer->col(i);}
    decltype(auto) col(long i) {return m_MatBuffer->col(i);}

    decltype(auto) row(long i) const {return m_MatBuffer->row(i);}
    decltype(auto) row(long i) {return m_MatBuffer->row(i);}

    //Unclear whether This should be the default. It is less efficient since it actually does the copy and likely wont be elided. Also cant be used to write to the matrix
    auto getCol(long i) const {return Matrix<T,sizei,1,Params,m_iIndexT,m_jIndexT>(m_MatBuffer->col(i),m_basis,m_spinSym);}
    auto getRow(long i) const {return Matrix<T,1,sizej,Params,m_iIndexT,m_jIndexT>(m_MatBuffer->row(i),m_basis,m_spinSym);}

    T& coeffRef(long i, long j){return m_MatBuffer.get().coeffRef(i,j);}
    T coeff(long i, long j)const {return m_MatBuffer.get().coeff(i,j);}
    T& operator()(long i, long j){return coeffRef(i,j);}
    const T& operator()(long i, long j) const {return m_MatBuffer.get()(i,j);}

    T& operator()(long i)
    {
        if constexpr(sizei == 1)
        {
            return m_MatBuffer.get()(0,i);
        }
        else if constexpr (sizej == 1)
        {
            return m_MatBuffer.get()(i,0);
        }
        else
            static_assert((sizei == 1 || sizej == 1) , "This method can only be called on vectors");
        //never reached
        return m_MatBuffer.get()(0,0);
    }
    const T& operator()(long i) const
    {
        if constexpr(sizei == 1)
        {
            return m_MatBuffer.get()(0,i);
        }
        else if constexpr (sizej == 1)
        {
            return m_MatBuffer.get()(i,0);
        }
        else
            static_assert((sizei == 1 || sizej == 1) , "This method can only be called on vectors");
        //never reached
        return m_MatBuffer.get()(0,0);
    }

    SparseMatrix<T,Params,m_iIndexT,m_jIndexT> sparseView() const
    {
        return SparseMatrix<T,Params,m_iIndexT,m_jIndexT>(m_MatBuffer->sparseView(),m_basis,m_spinSym);
    }

    //Returns a string or crashes if the two are different. Left for compatibility
    const std::string& getBasisS() const {releaseAssert(m_basis[0] == m_basis[1],"getBasisS not the same"); return m_basis[0];}
    const indexBasisT& getBasis() const {return m_basis;}

    //Interpret this as a basisChangeMatrix from m_Basis and add it to the Basis Manager
    void asBasisChangeMatrix() const
    {
        if constexpr(m_iIndexT == enums::IndexType::dual && m_jIndexT == enums::IndexType::direct)
            BasisManager::getInstance().addBasis(m_basis[1],m_basis[0],m_MatBuffer.get());
        else
            static_assert(dependent_false<decltype(*this)>::value , "ONLY DUAL-DIRECT MATRICES CAN BE BASIS CHANGE MATRICES");
    }

    //Interpret this as a basisChangeProjectionMatrix from m_Basis and add it to the Basis Manager
    void asProjectionMatrix() const
    {
        if constexpr(m_iIndexT == enums::IndexType::dual && m_jIndexT == enums::IndexType::direct)
            BasisManager::getInstance().addProjection(m_basis[1],m_basis[0],m_MatBuffer.get());
        else
            static_assert(dependent_false<decltype(*this)>::value , "ONLY DUAL-DIRECT MATRICES CAN BE PROJECTION MATRICES");
    }

    //Interpret this as a Metric and add it to the Basis Manager, Only valid for Hermitian Matrices
    void asMetric() const
    {
        releaseAssert(m_basis[0] == m_basis[1],"Basis for a metric cannot be different");
        if constexpr(m_iIndexT == enums::IndexType::improperDirect && m_jIndexT == enums::IndexType::direct && Params == enums::MatrixProperties::Hermitian)
            BasisManager::getInstance().addBasis(m_basis[0],m_MatBuffer.get());
        else
            static_assert(dependent_false<decltype(*this)>::value , "ONLY IMPROPERDIRECT-DIRECT HERMITIAN MATRICES CAN BE LOWERING METRICS");

    }

    //Interpret this as a basisChangeMatrix and add it to the Basis Manager as a temporary Basis
    [[nodiscard]] std::shared_ptr<std::string> asTemporaryBasisChangeMatrix() const
    {
        if constexpr(m_iIndexT == enums::IndexType::dual && m_jIndexT == enums::IndexType::direct)
            return BasisManager::getInstance().addTemporaryBasis(m_basis[0],m_MatBuffer.get());
        else
            static_assert(dependent_false<decltype(*this)>::value , "ONLY DUAL-DIRECT MATRICES CAN BE BASIS CHANGE MATRICES");
        return nullptr;// Never reached
    }
    //Interpret this as a Metric and add it to the Basis Manager as a temporary Basis
    [[nodiscard]] std::shared_ptr<std::string> asTemporaryMetric() const
    {
        releaseAssert(m_basis[0] == m_basis[1],"Basis for a metric cannot be different");
        if constexpr(m_iIndexT == enums::IndexType::improperDirect && m_jIndexT == enums::IndexType::direct && Params == enums::MatrixProperties::Hermitian)
            BasisManager::getInstance().addTemporaryBasis(m_MatBuffer.get());
        else
            static_assert(dependent_false<decltype(*this)>::value , "ONLY IMPROPERDIRECT-DIRECT HERMITIAN MATRICES CAN BE LOWERING METRICS");
        return nullptr;// Never reached
    }

    enums::SpinSymmetry getSpinSym() const {return m_spinSym;}
    void setSpinSym(enums::SpinSymmetry s) {m_spinSym = s;}
    void setBasis(const std::string& s) {setBasis({s,s});}
    void setBasis(const indexBasisT& s) {m_basis = s;}

    template<enums::MatrixProperties p, typename T2>
    explicit operator Matrix<T2,sizei,sizej,p,m_iIndexT,m_jIndexT>() const
    {//Explicit cast properties
        static_assert(std::is_same_v<T2,T> || std::is_same_v<T2,complexT>,"T2 in cast must be either complex T or T");
        if constexpr(std::is_same_v<T2,T>)
            return Matrix<T,sizei,sizej,p,m_iIndexT,m_jIndexT>(m_MatBuffer,m_basis,m_spinSym);
        else
            return Matrix<T2,sizei,sizej,p,m_iIndexT,m_jIndexT>(m_MatBuffer.get().template cast<T2>(),m_basis,m_spinSym);
    }

    operator T() const
    {
        static_assert(sizei == 1 && sizej == 1 , "Can only cast 1x1 matrices to scalars");
        return (*this)(0,0);
    }

    //TODO rows and cols are not necessary for a compile time construction
    static Matrix Identity(long rows, long cols,const std::string& basis, enums::SpinSymmetry sym){return Identity(rows,cols,{basis,basis},sym);}
    static Matrix Identity(long rows, long cols,const indexBasisT& basis, enums::SpinSymmetry sym)
    {
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        return Matrix(std::move(MatType::Identity(rows,cols)),basis,sym);
    }
    static Matrix Zero(long rows, long cols,const std::string& basis, enums::SpinSymmetry sym){return Zero(rows,cols,{basis,basis},sym);}

    static Matrix Zero(long rows, long cols,const indexBasisT& basis, enums::SpinSymmetry sym)
    {
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        return Matrix(std::move(MatType::Zero(rows,cols)),basis,sym);
    }
    //Until I figure out how to hide it better, this method should only be called from python. It is used to cast to numpy arrays. It is only implemented in the python module so will result in link errors. DO NOT IMPLEMENT
    MatType pyGetMatrixBuffer() const;

};

//Wraps an Eigen::SparseMatrix<T> and provides methods to compute the adjoint and basis changes automatically
template <typename T, enums::MatrixProperties Params,
         enums::IndexType m_iIndexT, enums::IndexType m_jIndexT>
class SparseMatrix
{
public:
    //Needed to be able to do the Header Hacks
    static constexpr int sizei = Eigen::Dynamic;
    static constexpr int sizej = Eigen::Dynamic;
    typedef T static_T;
    static constexpr int static_sizei = sizei;
    static constexpr int static_sizej = sizej;
    static constexpr enums::MatrixProperties static_Params = Params;
    static constexpr enums::IndexType static_iIndexT = m_iIndexT;
    static constexpr enums::IndexType static_jIndexT = m_jIndexT;
    typedef Eigen::SparseMatrix<T> MatType;
    typedef typename Eigen::NumTraits<T>::Real realT;
    typedef std::complex<realT> complexT;
    static constexpr bool isReal = std::is_same_v<realT,T>;
private:
    long m_rows = Eigen::Dynamic;
    long m_cols = Eigen::Dynamic;




    //Make every specialisation a friend
    template <typename, int, int, enums::MatrixProperties, enums::IndexType , enums::IndexType>
    friend class Matrix;
    template <typename, enums::MatrixProperties, enums::IndexType , enums::IndexType>
    friend class SparseMatrix;

    MatrixBufferObject<MatType> m_MatBuffer = MatrixBufferObject<MatType>::getEmpty();
    indexBasisT m_basis;
    enums::SpinSymmetry m_spinSym = enums::SpinSymmetry::NoSpin;

    SparseMatrix(const std::string& basis,enums::SpinSymmetry sym): SparseMatrix({basis,basis},sym){}

    SparseMatrix(const indexBasisT& basis,enums::SpinSymmetry sym)
    {
        //Allows setting spinSym and basis without giving it a buffer. Useful at times for checks
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        m_spinSym = sym;
        m_basis = basis;
    }

    SparseMatrix(const MatrixBufferObject<MatType>& buf, const std::string& basis,enums::SpinSymmetry sym):SparseMatrix(buf,{basis,basis},sym){}
    SparseMatrix(const MatrixBufferObject<MatType>& buf, const indexBasisT& basis,enums::SpinSymmetry sym)
    {
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        m_spinSym = sym;
        m_basis = basis;
        m_rows = buf->rows();
        m_cols = buf->cols();
        m_MatBuffer = buf;
    }



public:
    static constexpr bool isSparse = true;
    SparseMatrix(){}

    SparseMatrix(long rows, long cols, const std::string& basis, enums::SpinSymmetry sym) :SparseMatrix(rows,cols,{basis,basis},sym){}
    SparseMatrix(long rows, long cols, const indexBasisT& basis, enums::SpinSymmetry sym)
    {
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        resize(rows,cols,basis,sym);
    }
    SparseMatrix(MatType&& init, const std::string& basis, enums::SpinSymmetry sym) : SparseMatrix(std::move(init),{basis,basis},sym){}
    SparseMatrix(MatType&& init, const indexBasisT& basis, enums::SpinSymmetry sym) : m_MatBuffer()
    {
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        m_basis = basis;
        m_rows = init.rows();
        m_cols = init.cols();
        *m_MatBuffer = std::move(init);
        m_spinSym = sym;
    }
    SparseMatrix(const MatType& init, const std::string& basis, enums::SpinSymmetry sym) : SparseMatrix(init,{basis,basis},sym){}
    SparseMatrix(const MatType& init, const indexBasisT& basis, enums::SpinSymmetry sym) : m_MatBuffer()
    {
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        m_basis = basis;
        *m_MatBuffer = init;
        m_rows = init.rows();
        m_cols = init.cols();
        m_spinSym = sym;
    }
    //Init functions
    void setZero(long rows, long cols, const std::string& basis, enums::SpinSymmetry sym) {setZero(rows,cols,{basis,basis},sym);}
    void setZero(long rows, long cols, const indexBasisT& basis, enums::SpinSymmetry sym) {*this = std::move(Zero(rows,cols,basis,sym));}
    void setZero() {*this = std::move(Zero(m_rows,m_cols,m_basis,m_spinSym)); }
    void resize(long rows, long cols, const std::string& basis, enums::SpinSymmetry sym) {resize(rows,cols,{basis,basis},sym);}
    void resize(long rows, long cols, const indexBasisT& basis, enums::SpinSymmetry sym)
    {
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }

        if (m_MatBuffer.isEmpty())
            m_MatBuffer = MatrixBufferObject<MatType>();

        m_MatBuffer->resize(rows,cols);
        m_rows = rows;
        m_cols = cols;
        m_basis = basis;
        m_spinSym = sym;
    }

    //Operators
    //Does not support Eigen's lazy evaulation, yet
    friend std::ostream& operator<<(std::ostream& out, const SparseMatrix& o)
    {
        if (o.m_MatBuffer.isEmpty())
        {
            return out << "Write empty Matrix";
        }
        return out << o.m_MatBuffer.get();
    }
    friend class logger;
#define LINALG_FRIEND
#include "LinalgOperators.h"
#undef LINALG_FRIEND

    //Maths Operators

    auto diagonal() const
    {
        // #warning add back
        static_assert(checkCompileTimeTrace<T,sizei,sizej,MatrixMultiplicationParams<Params,Params>::v,m_iIndexT,m_jIndexT>::v , "Matrix Trace Check Failed");
        checkRunTimeTrace(m_rows,m_cols,m_basis);

        return m_MatBuffer->diagonal();
    }

    //Basis transformations
    //Returns false if not successful
    template <typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    [[nodiscard]] bool toBasis(SparseMatrix<T2, Params2,m_iIndexT2,m_jIndexT2>& dest, const std::string& desiredBasis,bool silenceSpinBlockWarning = false) const
    {return toBasis(dest,{desiredBasis,desiredBasis},silenceSpinBlockWarning);}

    template <typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    [[nodiscard]] bool toBasis(SparseMatrix<T2, Params2,m_iIndexT2,m_jIndexT2>& dest, const indexBasisT& desiredBasis,bool silenceSpinBlockWarning = false) const
    {

        typedef remove_cvref_t<decltype(dest)> destType;
        typedef remove_cvref_t<decltype(*this)> myType;

        static_assert(checkBasisReturn<T,sizei,sizej,Params,m_iIndexT,m_jIndexT,T2,sizei,sizej, Params2,m_iIndexT2,m_jIndexT2>::v , "Basis change compile time check failed");

        if constexpr(std::is_constructible_v<myType,destType>/* Evaluates to true if there is a cast from myType to destType*/)
        {
            if (m_basis == desiredBasis)
            {
                logger().log("Uneccessary Basis conversion");
                dest = static_cast<destType>(*this);
                return true;
            }
        }
        //If one is dynamic then they could be the same size and Eigen wont complain. However MatrixBufferObject will.
        // error: no match for ‘operator=’ (operand types are ‘MatrixBufferObject<Eigen::Matrix<std::complex<double>, -1, 1> >’ and ‘MatrixBufferObject<Eigen::Matrix<std::complex<double>, -1, -1> >’)
        constexpr bool sizeiIsSame = true;
        constexpr bool sizejIsSame = true;

        //If one of these is false then we have to assume the transformation is not the identity. If it is then we fail at runtime

        //Sort out the spin tags
        auto& BM = BasisManager::getInstance();
        bool validTags = false;
        auto desiredBasisiTags = BM.checkForTags(desiredBasis[0],&validTags);
        if (!validTags)
        {
            logger().log("Invalid Tags on basis", desiredBasis[0]);
            return false;
        }
        auto desiredBasisjTags = BM.checkForTags(desiredBasis[1],&validTags);
        if (!validTags)
        {
            logger().log("Invalid Tags on basis", desiredBasis[1]);
            return false;
        }

        auto myBasisiTags = BM.checkForTags(m_basis[0],&validTags);
        if (!validTags)
        {
            logger().log("Invalid Tags on basis", m_basis[0]);
            return false;
        }
        auto myBasisjTags = BM.checkForTags(m_basis[1],&validTags);
        if (!validTags)
        {
            logger().log("Invalid Tags on basis", m_basis[1]);
            return false;
        }

        bool IHaveA_i = BasisManager::contains(myBasisiTags,BasisManager::possibleTags::ABlock);
        bool IHaveB_i = BasisManager::contains(myBasisiTags,BasisManager::possibleTags::BBlock);

        bool IHaveA_j = BasisManager::contains(myBasisjTags,BasisManager::possibleTags::ABlock);
        bool IHaveB_j = BasisManager::contains(myBasisjTags,BasisManager::possibleTags::BBlock);

        bool desiredHasA_i = BasisManager::contains(desiredBasisiTags,BasisManager::possibleTags::ABlock);
        bool desiredHasB_i = BasisManager::contains(desiredBasisiTags,BasisManager::possibleTags::BBlock);

        bool desiredHasA_j = BasisManager::contains(desiredBasisjTags,BasisManager::possibleTags::ABlock);
        bool desiredHasB_j = BasisManager::contains(desiredBasisjTags,BasisManager::possibleTags::BBlock);

        if (IHaveA_i || IHaveA_j || IHaveB_i || IHaveB_j || desiredHasA_i || desiredHasA_j || desiredHasB_i || desiredHasB_j)
        {//Worry about spin
            //There are lots of headaches with using compile sized matrices here mainly due to getting their dimensions correct
            //at compile time but not knowing the dimensions at compile time as it depends on the spin transformation performed
            //for this reason we convert all compile time matrices to dynamic and call ourselves again, Nx1 vectors are fine
            // if constexpr(sizei == sizej && sizei != Eigen::Dynamic)
            // {
            //     typename changeMatrixArgs<Matrix,T,Eigen::Dynamic,Eigen::Dynamic>::Dense temp(m_MatBuffer.get(),m_basis,m_spinSym);
            //     return temp.toBasis(dest,desiredBasis);
            // }
            // else
            // {
            if (m_spinSym != enums::SpinSymmetry::RHF && m_spinSym != enums::SpinSymmetry::NoSpinSym)
                releaseAssert(false,"Requested a spin transformation for a matrix without spin");

            enum spinType
            {
                Alpha,
                Beta,
                Both,
            };

            spinType myLeftIndex = Both;
            spinType myRightIndex = Both;
            spinType desiredLeftIndex = Both;
            spinType desiredRightIndex = Both;

            if (IHaveA_i)
                myLeftIndex = Alpha;

            if (IHaveB_i)
                myLeftIndex = Beta;

            if (IHaveA_j)
                myRightIndex = Alpha;

            if (IHaveB_j)
                myRightIndex = Beta;

            if (desiredHasA_i)
                desiredLeftIndex = Alpha;

            if (desiredHasB_i)
                desiredLeftIndex = Beta;

            if (desiredHasA_j)
                desiredRightIndex = Alpha;

            if (desiredHasB_j)
                desiredRightIndex = Beta;

            BasisManager::possibleTags tagFrom_i = BasisManager::possibleTags::NullBasis;
            BasisManager::possibleTags tagFrom_j = BasisManager::possibleTags::NullBasis;
            BasisManager::possibleTags tagTo_i = BasisManager::possibleTags::NullBasis;
            BasisManager::possibleTags tagTo_j = BasisManager::possibleTags::NullBasis;

            std::string noSpinTagFrom_i = BasisManager::removeSpinTags(m_basis[0],&tagFrom_i);
            std::string noSpinTagFrom_j = BasisManager::removeSpinTags(m_basis[1],&tagFrom_j);

            std::string noSpinTagto_i = BasisManager::removeSpinTags(desiredBasis[0],&tagTo_i);
            std::string noSpinTagto_j = BasisManager::removeSpinTags(desiredBasis[1],&tagTo_j);
            //Not strictly necessary as CHECK does this warning too
            if (!silenceSpinBlockWarning && m_spinSym != enums::SpinSymmetry::RHF && (myLeftIndex != Both || myRightIndex != Both))
                logger().log("Requesting a Non RHF matrix come out of a spin block. This is probably not what you wanted");

            //If noSpinTagFrom and nospinTagto are in different sized bases to AND none of the matrices are dynamic then this will fail at runtime
            //as we have no way of knowing the size of *this before/after blocking
            //Further problems dest = 1x1 matrix, src = 2x2 matrix
            //transformi is never the identity in a well defined situation. e.g go to AA Block of 2x2 matrix.
            //But we cannot know this at compile time since the well definedness of the situation is determined at runtime
            //Therefore to avoid all these problems and more! We always use dynamic matrices here.

            //Pretend we always unblock and then block. If either one isnt actually a block then the check succeeds.
            typedef typename SpinUnBlocking<SparseMatrix,destType>::DenseResult IntermediateType;
            IntermediateType pseudoIntermediate1({noSpinTagFrom_i,noSpinTagFrom_j},m_spinSym);
            SpinUnBlocking<SparseMatrix,IntermediateType>::check(*this,pseudoIntermediate1,{tagFrom_i,tagFrom_j},silenceSpinBlockWarning);
            IntermediateType pseudoIntermediate2({noSpinTagto_i,noSpinTagto_j},m_spinSym);
            SpinBlocking<IntermediateType,destType>::check(pseudoIntermediate2,destType(desiredBasis,m_spinSym),{tagTo_i,tagTo_j});


            std::string leftFrom = noSpinTagFrom_i;
            if (myLeftIndex == Alpha)
                leftFrom += BasisManager::Alpha_Block;
            if (myLeftIndex == Beta)
                leftFrom += BasisManager::Beta_Block;

            std::string rightFrom = noSpinTagFrom_j;
            if (myRightIndex == Alpha)
                rightFrom += BasisManager::Alpha_Block;
            if (myRightIndex == Beta)
                rightFrom += BasisManager::Beta_Block;


            std::string leftTo = noSpinTagto_i;
            if (desiredLeftIndex == Alpha)
                leftTo += BasisManager::Alpha_Block;
            if (desiredLeftIndex == Beta)
                leftTo += BasisManager::Beta_Block;

            std::string rightTo = noSpinTagto_j;
            if (desiredRightIndex == Alpha)
                rightTo += BasisManager::Alpha_Block;
            if (desiredRightIndex == Beta)
                rightTo += BasisManager::Beta_Block;


            auto transformi = BM.getTransform_S(leftFrom,m_iIndexT,leftTo,m_iIndexT2);
            if (!transformi)
                return false;
            auto transformj = BM.getTransform_S(rightFrom,m_jIndexT,rightTo,m_jIndexT2);
            if (!transformj)
                return false;

            MatrixBufferObject<MatType> temp;
            MatrixBufferObject<typename destType::MatType> tempDest; // need to create a tempDest because dest may alias *this
            if (transformi.isIdentity || sizei == 1)
                *temp =  m_MatBuffer.get();
            else
                *temp = transformi->transpose() * m_MatBuffer.get();

            if (transformj.isIdentity || sizej == 1)
                *tempDest = temp;
            else
                *tempDest = temp.get()* *transformj;
            if (m_spinSym == enums::SpinSymmetry::RHF && (myLeftIndex != Both || myRightIndex != Both))
            {
                //Also have to pretend we have a beta_beta block
                leftFrom = noSpinTagFrom_i;
                if (myLeftIndex == Beta)
                    leftFrom += BasisManager::Alpha_Block;
                if (myLeftIndex == Alpha)
                    leftFrom += BasisManager::Beta_Block;

                rightFrom = noSpinTagFrom_j;
                if (myRightIndex == Beta)
                    rightFrom += BasisManager::Alpha_Block;
                if (myRightIndex == Alpha)
                    rightFrom += BasisManager::Beta_Block;

                auto transformi = BM.getTransform_S(leftFrom,m_iIndexT,leftTo,m_iIndexT2);
                if (!transformi)
                    return false;
                if (myLeftIndex == Both)
                    transformi.isIdentity = true;

                auto transformj = BM.getTransform_S(rightFrom,m_jIndexT,rightTo,m_jIndexT2);
                if (!transformi)
                    return false;
                if (myRightIndex == Both)
                    transformj.isIdentity = true;

                if (transformi.isIdentity || sizei == 1)
                    temp.get() =  m_MatBuffer.get();
                else
                    temp.get() = transformi->transpose() * m_MatBuffer.get();

                if (transformj.isIdentity || sizej == 1)
                    *tempDest += temp.get();
                else
                    *tempDest += temp.get()* *transformj;
            }
            dest = destType(tempDest,desiredBasis,m_spinSym);
            return true;
        }
        else
        {
            //No spin nonsense
            //For this reason the returned matrix should be used as v_mu = v_i Mat^i_mu
            auto transformi = BM.getTransform_S(m_basis[0],m_iIndexT,desiredBasis[0],m_iIndexT2);
            if (!transformi)
                return false;
            auto transformj = BM.getTransform_S(m_basis[1],m_jIndexT,desiredBasis[1],m_jIndexT2);
            if (!transformj)
                return false;

            MatrixBufferObject<MatType> temp;

            if (transformi.isIdentity || sizei == 1)
                temp.get() =  m_MatBuffer.get();
            else
                temp.get() = transformi->transpose() * m_MatBuffer.get();

            if (transformj.isIdentity || sizej == 1)
                dest = destType(temp,desiredBasis,m_spinSym);
            else
                dest = destType(temp.get()* *transformj,desiredBasis,m_spinSym);
            return true;
        }

    }

    template <typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    void toBasisC(SparseMatrix<T2, Params2,m_iIndexT2,m_jIndexT2>& dest, const std::string& desiredBasis) const {toBasisC(dest,{desiredBasis,desiredBasis});}

    //Same as toBasis but calls the check function automatically. Convenience
    template <typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    void toBasisC(SparseMatrix<T2, Params2,m_iIndexT2,m_jIndexT2>& dest, const indexBasisT& desiredBasis) const
    {
        bool success = toBasis(dest,desiredBasis);
        checkBasisChangeAssert(success,m_basis,desiredBasis);
    }



    //Error if not marked as self adjoint | enums::MatrixProperties::Hermitian

    [[nodiscard]] bool toHermitianBasis(typename HermReturnSparse<T,Params,m_iIndexT,m_jIndexT>::Mat& dest) const
    {
        releaseAssert(m_basis[0] == m_basis[1],"ToHermitian Basis with differing index basis");

        HermReturnSparse<T,Params,m_iIndexT,m_jIndexT>::check();
        if constexpr (m_iIndexT == enums::IndexType::dual && m_jIndexT == enums::IndexType::direct && Params == enums::MatrixProperties::Hermitian)
        {
            dest = *this;
            return true;
        }
        else
        {
            SparseMatrix<T,enums::MatrixProperties::None,m_iIndexT,m_jIndexT> temp;
            bool success = toBasis(temp,m_basis[0] + BasisManager::LowdinOrthogonaliseBasis);

            dest = static_cast<typename HermReturnSparse<T,Params,m_iIndexT,m_jIndexT>::Mat>(temp);
        }
    }

    void makeSelfAdjoint(bool multiplyBy2 = false, bool AllowInProjectedBasis = false)
    {
        releaseAssert(m_basis[0] == m_basis[1],"makeSelfAdjoint Basis with differing index basis");
        if (BasisManager::getInstance().isOrthogonal(m_basis[0]))
        {
            for (long i = 0; i < m_rows; i++)
            {
                for (long j = 0; j < i; j++)
                {
                    if constexpr(isReal)
                        (*this).coeffRef(j,i) = (*this).coeff(i,j);
                    else
                        (*this).coeffRef(j,i) = std::conj((*this).coeff(i,j));
                }
                if constexpr(!isReal)
                {
                    (*this).coeffRef(i,i) = (*this).coeff(i,i).real();
                }
            }
            if (multiplyBy2)
                *this *= 2;
        }
        else
        {
            *this += this->adjoint();
            if (!multiplyBy2)
                *this /=2;
        }
    }

    auto adjoint(bool AllowInProjectedBasis = false) const
    { // is the adjoint of a Unitary, Unitary - Probably
        if constexpr ((Params == enums::MatrixProperties::selfAdjoint) && indexContractionPair<m_jIndexT>::v == m_iIndexT)
        {
            releaseAssert(m_rows==m_cols,"Self Adjoint but not square matrix doesnt make sense");
            return *this;
        }
        else
        {
            if constexpr(Params == enums::MatrixProperties::Hermitian)
                if (BasisManager::getInstance().isOrthogonal(m_basis[0]) && BasisManager::getInstance().isOrthogonal(m_basis[1]))
                    return SparseMatrix<complexT,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v>(m_MatBuffer,{m_basis[1],m_basis[0]},m_spinSym);
            //Lower both indices if needed
            SparseMatrix<complexT,enums::MatrixProperties::None,indexConjugatePair<indexContractionPair<m_iIndexT>::v>::v,indexConjugatePair<indexContractionPair<m_jIndexT>::v>::v> partialAdjoint;
            toBasisC(partialAdjoint,m_basis);
            SparseMatrix<complexT,enums::MatrixProperties::None,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v> CanonicalAdjoint(partialAdjoint.m_MatBuffer->adjoint(),{m_basis[1],m_basis[0]},m_spinSym);

            return static_cast<SparseMatrix<complexT,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v>>(CanonicalAdjoint);
        }
    }

    auto conjugateTranspose() const
    {
        return SparseMatrix<T,Params,indexConjugatePair<m_jIndexT>::v,indexConjugatePair<m_iIndexT>::v>(m_MatBuffer->adjoint(),{m_basis[1],m_basis[0]},m_spinSym);
    }



    long rows() const {return m_rows;}
    long cols() const {return m_cols;}
    T coeff(long i, long j)const {return m_MatBuffer->coeff(i,j);}
    T& coeffRef(long i, long j){return m_MatBuffer->coeffRef(i,j);}
    void makeCompressed(){m_MatBuffer->makeCompressed();}

    template<enums::MatrixProperties p>
    explicit operator Matrix<T,Eigen::Dynamic,Eigen::Dynamic,p,m_iIndexT,m_jIndexT>() const
    {
        return Matrix<T,Eigen::Dynamic,Eigen::Dynamic,p,m_iIndexT,m_jIndexT>(m_MatBuffer->toDense(),m_basis,m_spinSym);
    }

    operator Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Params,m_iIndexT,m_jIndexT>() const
    {
        return Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Params,m_iIndexT,m_jIndexT>(m_MatBuffer->toDense(),m_basis,m_spinSym);
    }


    template<enums::MatrixProperties p>
    explicit operator SparseMatrix<T,p,m_iIndexT,m_jIndexT>() const
    {
        return SparseMatrix<T,p,m_iIndexT,m_jIndexT>(m_MatBuffer.get(),m_basis,m_spinSym);
    }

    realT norm() const
    {
        static_assert(checkCompileTimeMatrixMultiplication<complexT,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v
                                                           ,T,sizei,sizej,Params,m_iIndexT,m_jIndexT>::v , "Matrix Multiplication Check Failed ");
        checkRunTimeMatrixMultiplication(*this,*this);
        checkRunTimeTrace(m_rows,m_cols,m_basis);

        if constexpr (Params == enums::MatrixProperties::selfAdjoint)
            return m_MatBuffer->norm();
        else
        {
            if (!BasisManager::getInstance().isOrthogonal(m_basis[0]) || !BasisManager::getInstance().isOrthogonal(m_basis[1]))
                logger().log("Norm of Non self adjoint matrix can be slow");
            return std::sqrt(std::real(this->cdot(*this)));
        }
    }

    template <typename T2, enums::MatrixProperties Params2, enums::IndexType m_iIndexT2, enums::IndexType m_jIndexT2>
    auto cdot(const SparseMatrix<T2, Params2,m_iIndexT2,m_jIndexT2>& other) const
    {
        static_assert(checkCompileTimeMatrixMultiplication<complexT,Eigen::Dynamic,Eigen::Dynamic,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v
                                                           ,T2,Eigen::Dynamic,Eigen::Dynamic,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed ");
        static_assert(checkCompileTimeTrace<complexT,Eigen::Dynamic,Eigen::Dynamic,MatrixMultiplicationParams<Params,Params2>::v,indexContractionPair<m_jIndexT>::v,m_jIndexT2>::v , "Matrix Trace Check Failed");
        checkRunTimeMatrixMultiplication(*this,other);
        checkRunTimeTrace(m_rows,other.m_cols,{m_basis[0],other.m_basis[1]});

        if constexpr (Params == enums::MatrixProperties::selfAdjoint)
            return (m_MatBuffer.get() * other.m_MatBuffer.get()).toDense().trace();
        else
            if (BasisManager::getInstance().isOrthogonal(m_basis[0]) && BasisManager::getInstance().isOrthogonal(m_basis[1]))
                return (m_MatBuffer->adjoint() * other.m_MatBuffer.get()).toDense().trace();
            return (adjoint().m_MatBuffer.get() * other.m_MatBuffer.get()).toDense().trace();

    }

    const std::string& getBasisS() const {releaseAssert(m_basis[0] == m_basis[1],"getBasisS not the same"); return m_basis[0];}
    const indexBasisT& getBasis() const {return m_basis;}

    void setBasis(const std::string& s) {setBasis({s,s});}
    void setBasis(const indexBasisT& s) {m_basis = s;}

    enums::SpinSymmetry getSpinSym() const {return m_spinSym;}

    static SparseMatrix Zero(long rows, long cols,const std::string& basis, enums::SpinSymmetry sym){return Zero(rows,cols,{basis,basis},sym);}

    static SparseMatrix Zero(long rows, long cols,const indexBasisT& basis, enums::SpinSymmetry sym)
    {//Sparse Matrices are initialised to `zero' by default
        if constexpr (Params != enums::MatrixProperties::None)
        {
            releaseAssert(basis[0] == basis[1],"Matrices with indexes in different bases cannot be hermitian,selfadjoint,unitary etc");
        }
        return SparseMatrix(std::move(MatType(rows,cols)),basis,sym);
    }
};

#include "LinalgOperatorDef.h"
//Linalg depends on logger and so this must come afterwards.
//This may still lead to undefined references in the situation
/*
 * ComplexMatrix* ptr = getPtr();
 * logger().log("Mat",*ptr);
 */
//but only if no one has caused that particular log function to be instantiated. This is UNLIKELY...
template<typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1, enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
void logger::logAccurate(std::string name, const Matrix<T1, sizei1, sizej1, Params1, m_iIndexT1, m_jIndexT1> &mat)
{
    logAccurate(name,mat.m_MatBuffer.get());
}

template <typename T1, enums::MatrixProperties Params1,
         enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
void logger::logAccurate(std::string name, const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat)
{
    logAccurate(name,mat.m_MatBuffer.get());
}

#ifdef DOEXTERNTEMPLATES
#warning these are likely outdated
extern template class Matrix<numType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct>;
extern template class Matrix<numType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct>;

extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct>;
extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct>;
extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::direct>;

extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::improperDirect,enums::IndexType::direct>;
extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::improperDirect,enums::IndexType::direct>;
extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::selfAdjoint,enums::IndexType::improperDirect,enums::IndexType::direct>;

extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::improperDual>;
extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::improperDual>;
extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::improperDual>;


extern template class SparseMatrix<complexType,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::direct> ;
extern template class SparseMatrix<complexType,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct> ;
extern template class SparseMatrix<complexType,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::direct> ;

extern template class SparseMatrix<complexType,enums::MatrixProperties::None,enums::IndexType::improperDirect,enums::IndexType::direct>;
extern template class SparseMatrix<complexType,enums::MatrixProperties::Hermitian,enums::IndexType::improperDirect,enums::IndexType::direct>;
extern template class SparseMatrix<complexType,enums::MatrixProperties::selfAdjoint,enums::IndexType::improperDirect,enums::IndexType::direct>;

extern template class SparseMatrix<complexType,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::improperDual>;
extern template class SparseMatrix<complexType,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::improperDual>;
extern template class SparseMatrix<complexType,enums::MatrixProperties::selfAdjoint,enums::IndexType::dual,enums::IndexType::improperDual>;

extern template class Matrix<complexType,2,2,enums::MatrixProperties::Hermitian,enums::IndexType::dual,enums::IndexType::direct>;

extern template class Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::invalid>;
extern template class Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::direct,enums::IndexType::invalid>;
extern template class Matrix<complexType,Eigen::Dynamic,1,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::invalid>;
extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::eigenValue>;
extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::direct>;
extern template class Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::Unitary,enums::IndexType::dual,enums::IndexType::eigenValue>;
#endif



struct pauliMatrices
{
public:
    inline static const HermitianMatrix2cd sigmaX {Eigen::Matrix2Xcd {{0,1},{1,0}},"",enums::SpinSymmetry::SpinMatrix};
    inline static const HermitianMatrix2cd sigmaY {Eigen::Matrix2Xcd {{0,-1.*iu},{1.*iu,0}},"",enums::SpinSymmetry::SpinMatrix};
    inline static const HermitianMatrix2cd sigmaZ {Eigen::Matrix2Xcd {{1,0},{0,-1}},"",enums::SpinSymmetry::SpinMatrix};
    inline static const HermitianMatrix2cd sigmaI {Eigen::Matrix2Xcd {{1,0},{0,1}},"",enums::SpinSymmetry::SpinMatrix};
};


//Handles inf-inf = 0
inline numType subtractWithInfinities(numType A, numType B)
{
    if (A == posInf && B == posInf)
        return 0;
    if (A == negInf && B == negInf)
        return 0;
    return A-B;
}


inline bool imagLessThan(complexType a, complexType b)
{
    return (a.imag() < b.imag());
}

inline bool realLessThan(complexType a, complexType b)
{
    return (a.real() < b.real());
}



#endif // LINALG_H
