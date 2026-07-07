#include "greensfunction.h"
#include "phononselfenergy.h"
#include "quantitycalc.h"
#include "scfsolver.h"
#include "secondquantisedhamiltonian.h"
#include "simpleleadselfenergy.h"
#include "logger.h"
#include "EinSumTemplate.h"
#include <iostream>
#include <kiss_fft.h>
#include <memory>
#include <Eigen/LU>



logger stdoutlog;



int main()
{
    //Tensor rotation test
    ComplexMatrix Rot(3,3);
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


    SparseTensor<3,complexType> RotA;
    SparseTensor<3,complexType> RotA2;
    logger().log("A");
    logger().log(A.toString(0));
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,ri,rjk")(&A,&rotView,&RotA);
    logger().log("RotA");
    logger().log(RotA.toString(0));
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,ri,rjk")(&RotA,&InvrotView,&RotA2);
    logger().log("RotA2");
    logger().log(RotA2.toString(0));

    ComplexMatrix Id;
    SparseTensorView<ComplexMatrix,complexType,false> IdView(Id);
    Einsum<2,complexType,2,complexType,2,complexType>("ik,kj,ij")(&rotView,&InvrotView,&IdView);
    logger().log("id");
    std::cerr << Id << std::endl;
    logger().log("Rot");
    std::cerr << Rot<< std::endl;
    logger().log("RotInv");
    std::cerr << RotInv<< std::endl;

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

    SparseTensor<4,complexType> C1;
    Einsum<3,complexType,3,complexType,4,complexType>("kij,pqk,ijpq")(&APrep,&B,&C1);
    logger().log("APrep B C");
    logger().log(C1.toString(0));

    SparseTensor<4,complexType> C2;
    Einsum<3,complexType,3,complexType,4,complexType>("kij,pqk,ijpq")(&A,&B,&C2);
    logger().log("A B C");
    logger().log(C2.toString(0));
    releaseAssert(C1.isEqual(C2,0,true),"C1 != C2");


    //First only has repeats
    SparseTensor<2,complexType> Rot2;
    Rot2.copy(rotView);
    SparseTensor<2,complexType> RotPrep2;
    Rot2.prepareForEinsum<2,3,1,EinsumTemplates::firstType>("ij,ijk,k",RotPrep2);

    SparseTensor<1,complexType> D1;
    Einsum<2,complexType,3,complexType,1,complexType>("ij,ijk,k")(&RotPrep2,&A,&D1);
    logger().log("RotPrep2 A D");
    logger().log(D1.toString(0));

    SparseTensor<1,complexType> D2;
    Einsum<2,complexType,3,complexType,1,complexType>("ij,ijk,k")(&Rot2,&A,&D2);
    logger().log("A C D");
    logger().log(D2.toString(0));

    releaseAssert(D1.isEqual(D2,0,true),"D1 != D2");

    //First shares last index with the dest
    SparseTensor<3,complexType> APrep3;
    A.prepareForEinsum<3,2,1,EinsumTemplates::firstType>("ijk,ij,k",APrep3);

    Einsum<3,complexType,2,complexType,1,complexType>("ijk,ij,k")(&APrep3,&Rot2,&D1);
    logger().log("APrep3 A D");
    logger().log(D1.toString(0));

    Einsum<3,complexType,2,complexType,1,complexType>("ijk,ij,k")(&A,&Rot2,&D2);
    logger().log("A C D");
    logger().log(D2.toString(0));

    releaseAssert(D1.isEqual(D2,0,true),"D1 != D2, second");

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
    logger().log("`Opt' Basis Transform Levi-civita");
    logger().log(temp3.toString(0));

    Einsum<3,complexType,2,complexType,3,complexType>("ijk,ri,rjk")(&A,&rotView,&temp);
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,rj,irk")(&temp,&rotView,&temp2);
    Einsum<3,complexType,2,complexType,3,complexType>("ijk,rk,ijr")(&temp2,&rotView,&temp);
    logger().log("Basis Transform Levi-civita");
    logger().log(temp.toString(0));

    releaseAssert(temp3.isEqual(temp,0,true),"temp3 != temp");

    //Dot
    ComplexMatrix BigMatrix2(2000,2000);
    BigMatrix2.setRandom();

    auto start = std::chrono::high_resolution_clock::now();
    complexType innerProduct = (BigMatrix2.adjoint() * BigMatrix2).trace();
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
    ComplexMatrix BigMatrix(indexSize,indexSize);
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

    releaseAssert(Result1View.isEqual(Result2View),"BigTensor Failed");

















    return 0;

    logger().log("Passed Tests, now crash");
    releaseAssert(false,"Test Assert");




    // SparseTensor<3,double> D;
    // D.setSize(12);
    // D.coeffRef({0,5,0}) = 1;
    // D.coeffRef({4,0,0}) = 1;

    // SparseTensor<2,double> E;
    // Einsum<3,double,3,double,2,double>("mln,lpn,mp")(&D,&D,&E);
    // logger().log(E.toString());


    return 0;






    kiss_fft_cfg inverseCfg = kiss_fft_alloc( 1024 ,false ,0,0 );
    kiss_fft_cfg forwardCfg = kiss_fft_alloc( 1024 ,true ,0,0 );


    static_assert(std::is_same<kiss_fft_scalar,numType>::value);
    std::vector<kiss_fft_cpx> cx_in;
    std::vector<kiss_fft_cpx> cx_out;
    cx_in.resize(1024);
    cx_out.resize(1024);
    assert(1024 % 2 == 0);
    for (int i = 0; i < 1024; i++)
    {
        //cx_in[i] = {std::exp(2*M_PI*iu*(i/1024.)).real(),std::exp(2*M_PI*iu*(i/1024.)).imag()};
    }
    cx_in[1] = {1,0};

    kiss_fft( inverseCfg , &cx_in[0] , &cx_out[0] );
    //memset(&cx_out[numberOfPoints/2+1],0,sizeof(kiss_fft_cpx)*(numberOfPoints/2-1));
    kiss_fft( forwardCfg , &cx_out[0] , &cx_in[0] );
    for (int i = 0; i < 1024; i++)
    {
        cx_in[i].r /= 1024.;
        cx_in[i].i /= 1024.;
    }

    return 0;
}
