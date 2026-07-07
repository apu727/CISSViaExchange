#ifndef HAMILTONIANBASE_H
#define HAMILTONIANBASE_H

#include "linalgForwardDeclarations.h"
#include "sparsetensor.h"


class HamiltonianBase
{
public:


    HamiltonianBase();
    virtual ~HamiltonianBase(){};
    virtual const ComplexSelfAdjointSparseMatrix& getHamMatrix() = 0;
    virtual const ComplexSparseMatrix& getPhononCorrelationMatrix() = 0;
    virtual const SparseTensor<3,complexType>& getPhononGTensor() = 0;
    // virtual void save(std::string filename) const = 0;
    // virtual void load(FILE* file) = 0;
    virtual int getHilbertSpaceSize() = 0;
};

#endif // HAMILTONIANBASE_H
