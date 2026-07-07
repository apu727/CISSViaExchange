#include "PythonLinalg.hpp"

template void declare_Matrix<ComplexDualSelfAdjointMatrix AllMatrices>(py::module &,const std::string&);



void declare_linalg(py::module &m)
{
    declare_Matrix<RealMatrix AllMatrices>(m,"RealMatrix");
    declare_Matrix<RealHermitianMatrix AllMatrices>(m,"RealHermitianMatrix");

    declare_Matrix<ComplexMatrix AllMatrices>(m,"ComplexMatrix");
    declare_Matrix<ComplexHermitianMatrix AllMatrices>(m,"ComplexHermitianMatrix");
    declare_Matrix<ComplexSelfAdjointMatrix AllMatrices>(m,"ComplexSelfAdjointMatrix");

    declare_Matrix<ComplexDirectMatrix AllMatrices>(m,"ComplexDirectMatrix");
    declare_Matrix<ComplexDirectHermitianMatrix AllMatrices>(m,"ComplexDirectHermitianMatrix");
    declare_Matrix<ComplexDirectSelfAdjointMatrix AllMatrices>(m,"ComplexDirectSelfAdjointMatrix");

    declare_Matrix<ComplexDualMatrix AllMatrices>(m,"ComplexDualMatrix");
    declare_Matrix<ComplexDualHermitianMatrix AllMatrices>(m,"ComplexDualHermitianMatrix");
    declare_Matrix<ComplexDualSelfAdjointMatrix AllMatrices>(m,"ComplexDualSelfAdjointMatrix");

    declare_Sparse<ComplexSparseMatrix AllMatrices>(m,"ComplexSparseMatrix");
    declare_Sparse<ComplexHermitianSparseMatrix AllMatrices>(m,"ComplexHermitianSparseMatrix");
    declare_Sparse<ComplexSelfAdjointSparseMatrix AllMatrices>(m,"ComplexSelfAdjointSparseMatrix");

    declare_Sparse<ComplexDirectSparseMatrix AllMatrices>(m,"ComplexDirectSparseMatrix");
    declare_Sparse<ComplexDirectHermitianSparseMatrix AllMatrices>(m,"ComplexDirectHermitianSparseMatrix");
    declare_Sparse<ComplexDirectSelfAdjointSparseMatrix AllMatrices>(m,"ComplexDirectSelfAdjointSparseMatrix");

    declare_Sparse<ComplexDualSparseMatrix AllMatrices>(m,"ComplexDualSparseMatrix");
    declare_Sparse<ComplexDualHermitianSparseMatrix AllMatrices>(m,"ComplexDualHermitianSparseMatrix");
    declare_Sparse<ComplexDualSelfAdjointSparseMatrix AllMatrices>(m,"ComplexDualSelfAdjointSparseMatrix");

    //Aliases for cleaner code
    m.attr("RaisingMetric") = m.attr("ComplexDualHermitianMatrix");
    m.attr("LoweringMetric") = m.attr("ComplexDirectHermitianMatrix");
    m.attr("ForwardTransformMatrix") = m.attr("ComplexMatrix");
    m.attr("InverseTransformMatrix") = m.attr("ComplexMatrix");

    // declare_Matrix<RaisingMetric>(m,"RaisingMetric");
    // declare_Matrix<LoweringMetric>(m,"LoweringMetric");

    // declare_Matrix<ForwardTransformMatrix>(m,"ForwardTransformMatrix");
    // declare_Matrix<InverseTransformMatrix>(m,"InverseTransformMatrix");

    //Spin matrices generally
    declare_Matrix<HermitianMatrix2cd>(m,"HermitianMatrix2cd");
    declare_Matrix<UnitaryMatrix2cd>(m,"UnitaryMatrix2cd");

    //Vectors
    declare_Matrix<ComplexDualVector AllMatrices>(m,"ComplexDualVector");
    declare_Matrix<ComplexDirectVector AllMatrices>(m,"ComplexDirectVector");
    declare_Matrix<EigenValueVector AllMatrices>(m,"EigenValueVector");
    declare_Matrix<RealEigenValueVector>(m,"RealEigenValueVector");

    declare_Matrix<EigenValueMatrix AllMatrices>(m,"EigenValueMatrix");
    declare_Matrix<RealEigenValueMatrix AllMatrices>(m,"RealEigenValueMatrix");

    declare_Matrix<EigenVectorMatrix AllMatrices>(m,"EigenVectorMatrix");
    declare_Matrix<InvEigenVectorMatrix AllMatrices>(m,"InvEigenVectorMatrix");
    declare_Matrix<UnitaryEigenVectorMatrix AllMatrices>(m,"UnitaryEigenVectorMatrix");

    declare_Matrix<EigenVector AllMatrices>(m,"EigenVector");
    declare_Matrix<InvEigenVector AllMatrices>(m,"InvEigenVector");

    declare_Matrix<UnitaryEigenVector AllMatrices>(m,"UnitaryEigenVector");
    declare_Matrix<UnitaryInvEigenVector AllMatrices>(m,"UnitaryInvEigenVector");
}

