#include "EinSumTemplate.h"
#include "Tests/testutil.hpp"
#include "linalg.h"
#include "sparsetensor.h"

#include <chrono>
#include <Eigen/LU>

int einsumTests(int argc, char** argv)
{
    int ret = 0;

    //Tensor rotation test
    ComplexMatrix Rot(3,3,"",enums::SpinSymmetry::NoSpin);
    Rot.setZero();
    Rot.coeffRef(0,0) = cos(1);
    Rot.coeffRef(0,1) = -sin(1);
    Rot.coeffRef(1,0) = sin(1);
    Rot.coeffRef(1,1) = cos(1);
    Rot.coeffRef(2,2) = 1;
    ComplexMatrix RotInv = Rot.inverse();
    SparseTensorView<ComplexMatrix,complexType,false> rotView(Rot);
    SparseTensorView<ComplexMatrix,complexType,false> InvrotView(RotInv);

    SparseTensor<3,complexType> A;
    A.setSize(3);
    A.coeffRef({0,1,2}) = 1;
    A.coeffRef({1,2,0}) = 1;
    A.coeffRef({2,0,1}) = 1;
    A.coeffRef({1,0,2}) = -1;
    A.coeffRef({0,2,1}) = -1;
    A.coeffRef({2,1,0}) = -1;

    SparseTensor<3,complexType> RotAExpt;
    RotAExpt.setSize(3);
    RotAExpt.coeffRef({2,1,0}) = -1;
    RotAExpt.coeffRef({0,2,0}) = -0.8414709848078965;
    RotAExpt.coeffRef({1,2,0}) = 0.5403023058681398;
    RotAExpt.coeffRef({2,0,1}) = 1;
    RotAExpt.coeffRef({0,2,1}) = -0.5403023058681398;
    RotAExpt.coeffRef({1,2,1}) = -0.8414709848078965;
    RotAExpt.coeffRef({0,0,2}) = 0.8414709848078965;
    RotAExpt.coeffRef({1,0,2}) = -0.5403023058681398;
    RotAExpt.coeffRef({0,1,2}) = 0.5403023058681398;
    RotAExpt.coeffRef({1,1,2}) = 0.8414709848078965;


    SparseTensor<3,complexType> RotA;
    SparseTensor<3,complexType> RotA2;
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,ri,rjk")(&A,&rotView,&RotA);
    ret += expect(true,RotAExpt.isEqual(RotA,1e-16,true),__PRETTY_FUNCTION__,"RotA Failed");

    Einsum<3,complexType,2,complexType,3,complexType>("ijk,ri,rjk")(&RotA,&InvrotView,&RotA2);
    ret += expect(true,A.isEqual(RotA2,1e-15,true),__PRETTY_FUNCTION__,"InvRotA Failed");

    ComplexMatrix Id;
    SparseTensorView<ComplexMatrix,complexType,false> IdView(Id);
    Einsum<2,complexType,2,complexType,2,complexType>("ik,kj,ij")(&rotView,&InvrotView,&IdView);
    ComplexMatrix idExpt;
    idExpt.setIdentity(3,3,"",enums::SpinSymmetry::NoSpin);
    ret += expectNear(idExpt,Id,matrixNear,__PRETTY_FUNCTION__,"MatrixMultiplicationFailed");

    //Tensor prep test
    SparseTensor<3,complexType> APrep;
    A.prepareForEinsum<3,3,4,EinsumTemplates::firstType>("kij,pqk,ijpq",APrep);

    SparseTensor<3,complexType> B;
    B.setSize(3);
    B.coeffRef({0,1,2}) = 1;
    B.coeffRef({1,2,0}) = 1;
    B.coeffRef({2,0,1}) = 1;
    B.coeffRef({1,0,2}) = -1;
    B.coeffRef({0,2,1}) = -1;
    B.coeffRef({2,1,0}) = -1;

    SparseTensor<4,complexType> CExpt;
    CExpt.setSize(3);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int p = 0; p < 3; p++)
            {
                for (int q = 0; q < 3; q++)
                {
                    int v = 0;
                    v += (i == p && j == q);
                    v -= (i == q && j == p);
                    if (v != 0)
                        CExpt.coeffRef({i,j,p,q}) = v;
                }
            }
        }
    }


    SparseTensor<4,complexType> C1;
    Einsum<3,complexType,3,complexType,4,complexType>("kij,pqk,ijpq")(&APrep,&B,&C1);
    ret += expect(true,C1.isEqual(CExpt,0,true),__PRETTY_FUNCTION__,"APrep B C Failed");

    SparseTensor<4,complexType> C2;
    Einsum<3,complexType,3,complexType,4,complexType>("kij,pqk,ijpq")(&A,&B,&C2);
    ret += expect(true,C2.isEqual(CExpt,0,true),__PRETTY_FUNCTION__,"A B C Failed");


    //First only has repeats
    SparseTensor<2,complexType> Rot2;
    Rot2.copy(rotView);
    SparseTensor<2,complexType> RotPrep2;
    Rot2.prepareForEinsum<2,3,1,EinsumTemplates::firstType>("ij,ijk,k",RotPrep2);

    SparseTensor<1,complexType> DExpt;
    DExpt.setSize(3);
    DExpt.coeffRef({2}) = -0.8414709848078965;

    SparseTensor<1,complexType> D1;
    Einsum<2,complexType,3,complexType,1,complexType>("ij,ijk,k")(&RotPrep2,&A,&D1);
    ret += expect(true,D1.isEqual(DExpt,1e-15,true),__PRETTY_FUNCTION__,"RotPrep2 A D");

    SparseTensor<1,complexType> D2;
    Einsum<2,complexType,3,complexType,1,complexType>("ij,ijk,k")(&Rot2,&A,&D2);
    ret += expect(true,D2.isEqual(DExpt,1e-15,true),__PRETTY_FUNCTION__,"RotPrep2 A D");

    //First shares last index with the dest
    SparseTensor<3,complexType> APrep3;
    A.prepareForEinsum<3,2,1,EinsumTemplates::firstType>("ijk,ij,k",APrep3);

    Einsum<3,complexType,2,complexType,1,complexType>("ijk,ij,k")(&APrep3,&Rot2,&D1);
    ret += expect(true,D1.isEqual(DExpt,1e-15,true),__PRETTY_FUNCTION__,"APrep3 A D");

    Einsum<3,complexType,2,complexType,1,complexType>("ijk,ij,k")(&A,&Rot2,&D2);
    ret += expect(true,D2.isEqual(DExpt,1e-15,true),__PRETTY_FUNCTION__,"A C D");



    //basis transform of levi-civita tensor
    SparseTensor<3,complexType> temp;
    SparseTensor<3,complexType> temp2;
    SparseTensor<3,complexType> temp3;

    A.prepareForEinsum<3,2,3,EinsumTemplates::firstType>("ijk,ri,rjk",temp);
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,ri,rjk")(&temp,&rotView,&temp2);

    temp2.prepareForEinsum<3,2,3,EinsumTemplates::firstType>("ijk,rj,irk",temp);
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,rj,irk")(&temp,&rotView,&temp2);

    temp2.prepareForEinsum<3,2,3,EinsumTemplates::firstType>("ijk,rk,ijr",temp);
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,rk,ijr")(&temp,&rotView,&temp3);
    ret += expect(true,A.isEqual(temp3,1e-15,true),__PRETTY_FUNCTION__,"`Opt' Basis Transform Levi-civita");

    Einsum<3,complexType,2,complexType,3,complexType>("ijk,ri,rjk")(&A,&rotView,&temp);
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,rj,irk")(&temp,&rotView,&temp2);
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,rk,ijr")(&temp2,&rotView,&temp);
    ret += expect(true,A.isEqual(temp,1e-15,true),__PRETTY_FUNCTION__,"Basis Transform Levi-civita");

    //Dot
    ComplexMatrix BigMatrix2(2000,2000,"",enums::SpinSymmetry::NoSpin);
    BigMatrix2.setRandom();

    auto start = std::chrono::high_resolution_clock::now();
    complexType innerProduct = BigMatrix2.norm();//(BigMatrix2.adjoint() * BigMatrix2).trace();
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    logger().log("BigMatrix dot us:",duration.count());
    logger().log("BigMatrix dot val:",innerProduct);

    start = std::chrono::high_resolution_clock::now();
    complexType norm = BigMatrix2.squaredNorm();
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    logger().log("BigMatrix dot us:",duration.count());
    logger().log("BigMatrix dot val:",norm);


    //Test a big Tensor
    logger().log("Start Constructing Big Tensor");
    int indexSize = 100;
    DenseTensor<4,complexType> BigTensor(indexSize);// has at most 10^8 elements
    DenseTensor<4,complexType>::indexArrayType idxs;
    for (int i = 0; i < (int)1e8; i++)
    {
        for (int idxPos =0; idxPos < 4; idxPos++)
        {
            idxs[idxPos] = rand() % indexSize;
        }
        BigTensor.coeffRef(idxs) += rand();
        if (i%(int)1e6 == 0)
            logger().log("On: ",i);
    }
    ComplexMatrix BigMatrix(indexSize,indexSize,"",enums::SpinSymmetry::NoSpin);
    BigMatrix.setRandom();

    ComplexMatrix Result1;
    ComplexMatrix Result2;
    SparseTensorView<ComplexMatrix,complexType,false> BigMatrixView(BigMatrix);
    SparseTensorView<ComplexMatrix,complexType,false> Result1View(Result1);
    SparseTensorView<ComplexMatrix,complexType,false> Result2View(Result2);

    SparseTensor<4,complexType> BigTensorPrep;
    logger().log("Done Constructing Big Tensor");
    start = std::chrono::high_resolution_clock::now();
    BigTensor.prepareForEinsum<4,2,2,EinsumTemplates::firstType>("pqrs,rp,qs",BigTensorPrep,true);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    logger().log("Time to prep us:",duration.count());

    start = std::chrono::high_resolution_clock::now();
    Einsum<4,complexType,2,complexType,2,complexType>("pqrs,rp,qs")(&BigTensorPrep,&BigMatrixView,&Result1View);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    logger().log("Prepped Time us:",duration.count());

    start = std::chrono::high_resolution_clock::now();
    Einsum<4,complexType,2,complexType,2,complexType>("pqrs,rp,qs")(&BigTensor,&BigMatrixView,&Result2View);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    logger().log("Not Prepped Time us:",duration.count());

    ret += expect(true,Result1View.isEqual(Result2View),__PRETTY_FUNCTION__,"BigTensor Failed");

    return ret > 0 ? -1 : 0;
}

