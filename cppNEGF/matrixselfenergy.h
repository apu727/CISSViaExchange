#ifndef MATRIXSELFENERGY_H
#define MATRIXSELFENERGY_H


#include "secondquantisedhamiltonian.h"
#include "selfenergybase.h"
template<class SelfEnergyMatrixType, bool selfAdjoint = true>
class MatrixSelfEnergy : public HamiltonianBase, public SelfEnergyBase<SelfEnergyMatrixType>
{
public:
    using SparseMatrixType = std::conditional_t<selfAdjoint,ComplexSelfAdjointSparseMatrix,ComplexSparseMatrix>;
    using DenseMatrixType = std::conditional_t<selfAdjoint,ComplexSelfAdjointMatrix,ComplexMatrix>;
private:

    SparseMatrixType m_MatrixProvided;// This may be in a different basis to what is needed

    SelfEnergyMatrixType m_SigmaR; // we can do this because its a constant
    SelfEnergyMatrixType m_SigmaA; // we can do this because its a constant
    SparseMatrixType m_SigmaRS; // needed to satisfy HamiltonianBase requirements
    SparseMatrixType m_SigmaAS; // needed to satisfy HamiltonianBase requirements
    indexBasisT m_workingBasis = {};

public:

    MatrixSelfEnergy(const SparseMatrixType& Matrix);
    MatrixSelfEnergy(const DenseMatrixType& Matrix);
    MatrixSelfEnergy(SecondQuantisedHamiltonian& Matrix);
    virtual ~MatrixSelfEnergy(){};

    //Hamiltonian Virtual Functions
    //Dont really want to change the interface code for a function that isnt called. Hence runtime disable this function if Ham is not self adjoint. Use getSigmaR.
    virtual const ComplexSelfAdjointSparseMatrix& getHamMatrix() override {if constexpr(SparseMatrixType::static_Params == enums::MatrixProperties::selfAdjoint) return m_SigmaRS; else __builtin_trap();}
    virtual const ComplexSparseMatrix& getPhononCorrelationMatrix()override{__builtin_trap();}
    virtual const SparseTensor<3,complexType>& getPhononGTensor()override{__builtin_trap();}
    virtual int getHilbertSpaceSize()override{return m_SigmaR.rows();}

    //Self Energy Virtual Functions
    virtual void getSigmaIn(numType E, SelfEnergyMatrixType& dest) override{dest.setZero(m_SigmaR.rows(),m_SigmaR.cols(),m_workingBasis,enums::SpinSymmetry::RHF);}
    //virtual void getSigmaIn(SelfEnergyMatrixType& dest) override {return getSigmaIn(0,dest);};

    virtual void getSigmaOut(numType E, SelfEnergyMatrixType& dest) override{dest.setZero(m_SigmaR.rows(),m_SigmaR.cols(),m_workingBasis,enums::SpinSymmetry::RHF);}
    //virtual void getSigmaOut(SelfEnergyMatrixType& dest) override {return getSigmaOut(0,dest);};

    virtual void getSigmaR(complexType E, SelfEnergyMatrixType& dest) override{dest = m_SigmaR;}
    virtual void getSigmaR(SelfEnergyMatrixType& dest) override {return getSigmaR(0,dest);}

    virtual void getSigmaA(complexType E, SelfEnergyMatrixType& dest) override{dest = m_SigmaA;}
    virtual void getSigmaA(SelfEnergyMatrixType& dest) override {return getSigmaA(0,dest);}

    virtual void getSigmaK(numType E, SelfEnergyMatrixType& dest) override;


    // virtual void save(std::string filename) const override{logger().log("not Implemented");}
    // virtual void load(FILE* file) override{logger().log("not Implemented");}
    virtual bool hasSigmaK() override {return !selfAdjoint;}
    virtual const indexBasisT& getBasis() override{return m_workingBasis;}
    virtual void setBasis(const std::string& bas) override;
    virtual void setBasis(const indexBasisT& bas) override;

};

#endif // MATRIXSELFENERGY_H
