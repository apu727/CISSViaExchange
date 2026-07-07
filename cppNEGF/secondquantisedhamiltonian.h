#ifndef SECONDQUANTISEDHAMILTONIAN_H
#define SECONDQUANTISEDHAMILTONIAN_H

#include "hamiltonianbase.h"
#include "linalg.h"


class SecondQuantisedHamiltonian : public HamiltonianBase
{
    ComplexSelfAdjointSparseMatrix m_HamMatrix;
    ComplexSparseMatrix m_PhononCorrelationMatrix;
    SparseTensor<3,complexType> m_PhononGTensor;

    bool m_areMatricesCached = false;
    complexType computeHamiltonianMatrixElementFromParameters(std::vector<char>& ket1, std::vector<char>& ket2);
    void computeMatricesFromParameters();
    complexType computePhononCorrelationMatrixElementFromParameters(std::vector<char>& ket1, std::vector<char>& ket2);
    complexType computePhononGTensorElementFromParameters(std::vector<char>& ket1, std::vector<char>& ket2, int phononPos);

    void computeMatrices();

    enum loadingType
    {
        None = 0,
        fromParameters,
        fromFile
    } m_loadingType;



public:
    typedef struct Parameters
    {
        std::vector<complexType> U_0Alpha; // on site interaction
        std::vector<complexType> U_0Beta; // on site interaction
        numType U_1 = 0; // on site interaction phonon
        numType t_0 = 0; // nearest neighbour hopping
        numType lambda_0 = 0; // next nearest neighbour hopping (Spin orbit)
        numType t_1 = 0; // phonon assisted hopping
        numType lambda_1 = 0; // phonon assisted next nearest neighbour hopping (Spin orbit)
        numType meff = 1; // ratio of me
        numType R = 1;  // radius of loops
        numType P = 1; // pitch of loops
        numType N = 1; // number of loops
        numType M = 1; // sites per loop
        int HilbertSpaceSize = 1;
        int numberOfElectrons = 1; // only 1 is supported ATM
        std::string basis = "";
    } Parameters;

    SecondQuantisedHamiltonian();
    SecondQuantisedHamiltonian(const Parameters& params){loadParameters(params);};
    SecondQuantisedHamiltonian(const std::string& filename){loadFile(filename);};
    virtual ~SecondQuantisedHamiltonian(){};

    void loadParameters(const Parameters& params);;
    void loadFile(const std::string& filename);

    const ComplexSelfAdjointSparseMatrix& getHamMatrix() override {if (!m_areMatricesCached) computeMatrices(); return m_HamMatrix;}
    virtual const ComplexSparseMatrix& getPhononCorrelationMatrix() override {if (!m_areMatricesCached) computeMatrices(); return m_PhononCorrelationMatrix;};
    virtual const SparseTensor<3,complexType>& getPhononGTensor() override {if (!m_areMatricesCached) computeMatrices(); return m_PhononGTensor;};
    // virtual void save(std::string filename) const override;
    // virtual void load(FILE* file) override;
    virtual int getHilbertSpaceSize() override {return m_params.HilbertSpaceSize;}
private:
    Parameters m_params;
};

#endif // SECONDQUANTISEDHAMILTONIAN_H
