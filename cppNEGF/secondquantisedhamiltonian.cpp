#include "secondquantisedhamiltonian.h"
#include "configfileloader.h"
#include "logger.h"

#include "Eigen/Geometry"



complexType SecondQuantisedHamiltonian::computeHamiltonianMatrixElementFromParameters(std::vector<char>& ket1, std::vector<char>& ket2)
{/* Computes \braket{H_\t{mol}} between 2 kets in the Hilbert space for the molecule ket1 and ket2 are two bitstrings (fermions)*/
    int numberOfSpinOrbitals = ket1.size(); // Encoding is alpha|beta|alpha|beta
    complexType Energy = 0;
    //std::vector< epsilon_m = m_params.U_0 + 2*m_params.t_0;// - m_params.t_0/4*iu;

    auto spiralLocation = [CONST_REF_CAPTURE(m_params)](int m)
    {
        auto& R = m_params.R;
        auto& M = m_params.M;
        auto& P = m_params.P;
        return Eigen::Vector3d(R*std::cos(m*2*M_PI/M),R*std::sin(m*2*M_PI/M),m*P/M);
    };
    std::vector<int> differences;
    std::vector<int> braOccDifferences;
    std::vector<int> ketOccDifferences;
    for (int i = 0; i < numberOfSpinOrbitals; i++)
    {
        if (ket1[i] != ket2[i])
        {
            differences.push_back(i);
            if (ket1[i])
                braOccDifferences.push_back(i);
            if (ket2[i])
                ketOccDifferences.push_back(i);
        }
    }
    if (differences.size() == 0)
    {    /* no Excitation */
        for (int i = 0; i < numberOfSpinOrbitals; i++)
        {
            if (ket1[i])
            {
                Energy += (i%2 == 1? m_params.U_0Beta[i/2] : m_params.U_0Alpha[i/2]) + 2*m_params.t_0;
                // if (i == 0 || i == 1)
                //     Energy -= m_params.t_0;
                // if (i == numberOfSpinOrbitals-1 || i == numberOfSpinOrbitals-2)
                //     Energy -= m_params.t_0;
            }
        }
    }
    else if (differences.size() == 2)
    {/* Single Excitation */
        char spinBra = braOccDifferences[0]%2;
        char spinKet = ketOccDifferences[0]%2;
        //\psi^\dagger_m \Psi_{m+1}
        if (std::abs(differences[0]-differences[1]) == 2)
        {
            Energy -= m_params.t_0;
            // if (braOccDifferences[0] > ketOccDifferences[0])
            //     Energy += -iu*m_params.lambda_0*pauliMatrices.sigmaZ(spinBra,spinKet);
            // else
            //     Energy += iu*m_params.lambda_0*pauliMatrices.sigmaZ(spinBra,spinKet);
        }
        if (std::abs(braOccDifferences[0]-spinBra - (ketOccDifferences[0]-spinKet)) == 4)
        { //Spin orbit hopping (next-nearest neighbour)
            int spatialm = std::min(differences[0],differences[1])/2;
            assert(spatialm+2 < numberOfSpinOrbitals/2);
            Eigen::Vector3d rm = spiralLocation(spatialm);
            Eigen::Vector3d rm1 = spiralLocation(spatialm+1);
            Eigen::Vector3d rm2 = spiralLocation(spatialm+2);
            Eigen::Vector3d vm = ((rm-rm1).normalized()).cross((rm1-rm2).normalized());

            HermitianMatrix2cd Spinmatrix;
            pauliMatrices::sigmaX*vm[0];
            Spinmatrix = pauliMatrices::sigmaX * vm[0] + pauliMatrices::sigmaY*vm[1] + pauliMatrices::sigmaZ*vm[2];
            //We implement: \iu a^\dagger_m v^(+)_m\cdot \sigma a_{m+2}  -\iu a^\dagger_{m+2} v^(+)_m\cdot \sigma a_{m}
            //Alternatively \iu a^\dagger_m v^(+)_m\cdot \sigma a_{m+2}  +\iu a^\dagger_{m+2} v^(-)_{m+2}\cdot \sigma a_{m} is equivalent
            if (braOccDifferences[0] > ketOccDifferences[0])
                Energy += -iu*m_params.lambda_0*Spinmatrix(spinBra,spinKet);
            else
                Energy += iu*m_params.lambda_0*Spinmatrix(spinBra,spinKet);
        }
    }
    return Energy;
}

void SecondQuantisedHamiltonian::computeMatricesFromParameters()
{
    int HilbertSpaceSize = m_params.HilbertSpaceSize;
    m_HamMatrix = ComplexSelfAdjointSparseMatrix(HilbertSpaceSize,HilbertSpaceSize,m_params.basis,enums::SpinSymmetry::NoSpinSym);
    m_PhononCorrelationMatrix = ComplexSparseMatrix(HilbertSpaceSize,HilbertSpaceSize,m_params.basis,enums::SpinSymmetry::NoSpinSym);
    assert(HilbertSpaceSize %2 == 0);
    m_PhononGTensor.setSize(HilbertSpaceSize);
    std::vector<char> ket1(HilbertSpaceSize,0);
    std::vector<char> ket2(HilbertSpaceSize,0);

    for (int i = 0; i < HilbertSpaceSize; i++)
    {
        ket1[i] = 1;
        for (int j = 0; j < HilbertSpaceSize; j++)
        {
            ket2[j] = 1;
            complexType E = computeHamiltonianMatrixElementFromParameters(ket1,ket2);
            if (E != complexType(0))
                m_HamMatrix.coeffRef(i,j) = E;

            complexType E2 = computePhononCorrelationMatrixElementFromParameters(ket1,ket2);
            if (E2 != complexType(0))
                m_PhononCorrelationMatrix.coeffRef(i,j) = E2;
            for (int n = 0; n < HilbertSpaceSize/2; n++)
            {

                complexType E3 = computePhononGTensorElementFromParameters(ket1,ket2,n);
                if (E3 != complexType(0))
                    m_PhononGTensor.coeffRef({i,j,n}) = E3;
            }
            ket2[j] = 0;

        }
        ket1[i] = 0;
    }
}

complexType SecondQuantisedHamiltonian::computePhononCorrelationMatrixElementFromParameters(std::vector<char> &ket1, std::vector<char> &ket2)
{/*Computes \braket{H_{e-ph}} between 2 kets in the Hilbert space for the molecule
       Essentially computes the terms t_1 a^\dagger_m+1 a_m (b_m + b_m^\dagger) for use in the phonon self energy. computes g_r,r'?*/
    int numberOfSpinOrbitals = ket1.size(); // Encoding is alpha|beta|alpha|beta
    complexType Energy = 0;

    auto spiralLocation = [CONST_REF_CAPTURE(m_params)](int m)
    {
        auto& R = m_params.R;
        auto& M = m_params.M;
        auto& P = m_params.P;
        return Eigen::Vector3d(R*std::cos(m*2*M_PI/M),R*std::sin(m*2*M_PI/M),m*P/M);
    };
    auto SOSpinMatrix = [BY_VAL_CAPTURE(numberOfSpinOrbitals),CONST_REF_CAPTURE(spiralLocation)](int m, int s = 1)
    {
        //compute v_m^{(s)} = d_{m+s} \times d_{m+2s}
        //d_{m+s} = (r_m-r_{m+s})/|r_m-r_{m+s}|
        s = s > 0 ? 1 : -1;
        assert(m+2*s < numberOfSpinOrbitals/2 && m < numberOfSpinOrbitals/2);
        assert(m+2*s >= 0 && m >= 0);

        Eigen::Vector3d rm = spiralLocation(m);
        Eigen::Vector3d rm1 = spiralLocation(m+s);
        Eigen::Vector3d rm2 = spiralLocation(m+2*s);
        Eigen::Vector3d vm = ((rm-rm1).normalized()).cross((rm1-rm2).normalized());

        HermitianMatrix2cd Spinmatrix;

        Spinmatrix = pauliMatrices::sigmaX*vm[0] + pauliMatrices::sigmaY*vm[1] + pauliMatrices::sigmaZ*vm[2];
        return Spinmatrix;
    };
    auto ValidateSOBraket = [BY_VAL_CAPTURE(numberOfSpinOrbitals)](int m, int s)
    {// validates calling SOSpinMatrix. Avoids the edge conditions
        bool valid = true;
        valid &= (m+2*s < numberOfSpinOrbitals/2 && m < numberOfSpinOrbitals/2);
        valid &= (m+2*s >= 0 && m >= 0);
        return valid;
    };

    int braOcc = -1;
    int ketOcc = -1;
    char spinBra = -1;
    char spinKet = -1;

    for (int i = 0; i < numberOfSpinOrbitals; i++)
    {
        if (ket1[i])
            braOcc = i;
        if (ket2[i])
            ketOcc = i;
    }
    spinBra = braOcc % 2;
    spinKet = ketOcc % 2;

    int spatialBra = braOcc/2; // integer divide
    int spatialKet = ketOcc/2; // integer divide
    // \sum_{s=\pm 1} \sum_{s'=\pm 1} t_1^2 G_{m+s+s',n} - \lambda_1^2 (v_m^{(s)}\cdot\sigma)(v_{m+2s}^{(s')}\cdot\sigma)G_{m+2(s+s'),n}
    // - \iu t_1 \lambda_1 (\sigma\cdot v_m^{(s)})G_{m+2s+s',n} - \iu t_1 \lambda_1(\sigma\cdot v_{m+s+2s'}^{(s')})G_{m+s+2s',n}
    for (int s = -1; s <= 1; s += 2)
    {
        for (int sPrime = -1; sPrime <= 1; sPrime += 2)
        {
            if (spatialBra + s + sPrime == spatialKet)
            {
                Energy += m_params.t_1*m_params.t_1;
            }
            if (spatialBra + 2*(s+sPrime) == spatialKet && ValidateSOBraket(spatialBra,s) && ValidateSOBraket(spatialBra+2*s,sPrime))
            {
                HermitianMatrix2cd Spinmatrix1 = SOSpinMatrix(spatialBra,s);
                HermitianMatrix2cd Spinmatrix2 = SOSpinMatrix(spatialBra+2*s,sPrime);
                Energy -= m_params.lambda_1*m_params.lambda_1*(Spinmatrix1*Spinmatrix2)(spinBra,spinKet);
            }
            if (spatialBra + 2*s+sPrime == spatialKet && ValidateSOBraket(spatialBra,s))
            {
                HermitianMatrix2cd Spinmatrix1 = SOSpinMatrix(spatialBra,s);
                Energy -= iu*m_params.t_1*m_params.lambda_1*Spinmatrix1(spinBra,spinKet);
            }
            if (spatialBra + s+2*sPrime == spatialKet && ValidateSOBraket(spatialBra+s,sPrime))
            {
                HermitianMatrix2cd Spinmatrix1 = SOSpinMatrix(spatialBra+s,sPrime);
                Energy -= iu*m_params.t_1*m_params.lambda_1*Spinmatrix1(spinBra,spinKet);
            }
        }
    }

    return Energy; // negate this?
    //return -Energy; // negate this?
}
complexType SecondQuantisedHamiltonian::computePhononGTensorElementFromParameters(std::vector<char> &ket1, std::vector<char> &ket2, int phononPos)
{/*Computes \braket{G_{e-ph}} between 2 kets in the Hilbert space for the molecule
       Essentially computes the terms t_1 a^\dagger_m+1 a_m (b_m + b_m^\dagger) for use in the phonon self energy. computes g_r,r'?*/
    int numberOfSpinOrbitals = ket1.size(); // Encoding is alpha|beta|alpha|beta
    assert(phononPos < numberOfSpinOrbitals/2);
    phononPos = 2*phononPos;
    complexType Energy = 0;
    numType epsilon_m = m_params.U_1;//+ 2*m_params.t_0;

    auto spiralLocation = [CONST_REF_CAPTURE(m_params)](int m)
    {
        auto& R = m_params.R;
        auto& M = m_params.M;
        auto& P = m_params.P;
        return Eigen::Vector3d(R*std::cos(m*2*M_PI/M),R*std::sin(m*2*M_PI/M),m*P/M);
    };
    std::vector<int> differences;
    std::vector<int> braOccDifferences;
    std::vector<int> ketOccDifferences;
    for (int i = 0; i < numberOfSpinOrbitals; i++)
    {
        if (ket1[i] != ket2[i])
        {
            differences.push_back(i);
            if (ket1[i])
                braOccDifferences.push_back(i);
            if (ket2[i])
                ketOccDifferences.push_back(i);
        }
    }
    if (differences.size() == 0)
    {    /* no Excitation */
        if (ket1[phononPos] || ket1[phononPos+1])
            Energy += epsilon_m;
    }
    else if (differences.size() == 2)
    {/* Single Excitation */
        char spinBra = braOccDifferences[0]%2;
        char spinKet = ketOccDifferences[0]%2;
        //\psi^\dagger_m \Psi_{m+1}
        if (spinBra == spinKet && ((phononPos+spinKet+2 < numberOfSpinOrbitals && ket1[phononPos+spinBra] && ket2[phononPos+spinKet+2]) || (phononPos+spinBra+2 < numberOfSpinOrbitals && ket1[phononPos+spinBra+2] && ket2[phononPos+spinKet])))
        {
            Energy -= m_params.t_1;
        }
        if (std::abs(braOccDifferences[0]-spinBra - (ketOccDifferences[0]-spinKet)) == 4)
        { //Spin orbit hopping (next-nearest neighbour)
            int spatialm = std::min(differences[0],differences[1])/2;
            if (spatialm != phononPos/2)
                return Energy;

            assert(spatialm+2 < numberOfSpinOrbitals/2);
            Eigen::Vector3d rm = spiralLocation(spatialm);
            Eigen::Vector3d rm1 = spiralLocation(spatialm+1);
            Eigen::Vector3d rm2 = spiralLocation(spatialm+2);
            Eigen::Vector3d vm = ((rm-rm1).normalized()).cross((rm1-rm2).normalized());

            HermitianMatrix2cd Spinmatrix;

            Spinmatrix = pauliMatrices::sigmaX*vm[0] + pauliMatrices::sigmaY*vm[1] + pauliMatrices::sigmaZ*vm[2];
            //We implement: \iu a^\dagger_m v^(+)_m\cdot \sigma a_{m+2}  -\iu a^\dagger_{m+2} v^(+)_m\cdot \sigma a_{m}
            //Alternatively \iu a^\dagger_m v^(+)_m\cdot \sigma a_{m+2}  +\iu a^\dagger_{m+2} v^(-)_{m+2}\cdot \sigma a_{m} is equivalent
            if (braOccDifferences[0] > ketOccDifferences[0])
                Energy += -iu*m_params.lambda_1*Spinmatrix(spinBra,spinKet);
            else
                Energy += iu*m_params.lambda_1*Spinmatrix(spinBra,spinKet);
        }
    }
    return Energy;
}

void SecondQuantisedHamiltonian::computeMatrices()
{
    if (m_loadingType == fromParameters)
    {
        computeMatricesFromParameters();
    }
    else if (m_loadingType == fromFile)
    {
        assert(false);
    }
    else
    {
        assert(false);
    }
    m_areMatricesCached = true;
}

SecondQuantisedHamiltonian::SecondQuantisedHamiltonian()
{
    m_loadingType = None;
}

void SecondQuantisedHamiltonian::loadParameters(const Parameters &params)
{
    logger().log("LengthUnit",lengthUnit);
    m_params = params;
    while(m_params.U_0Alpha.size() <= m_params.HilbertSpaceSize/2)
    {
        m_params.U_0Alpha.push_back(0);
    }
    while(m_params.U_0Beta.size() <= m_params.HilbertSpaceSize/2)
    {
        m_params.U_0Beta.push_back(0);
    }


    m_loadingType = fromParameters;
}

void SecondQuantisedHamiltonian::loadFile(const std::string &filename)
{
    m_loadingType = fromFile;
    assert(false);
}

// void SecondQuantisedHamiltonian::save(std::string filename) const
// {
//     logger logFile = logger(filename,true);
//     logFile.log("SecondQuantisedHamiltonian::Parameters\n{");
//     logFile.log("U_0Alpha",m_params.U_0Alpha);
//     logFile.log("U_0Bea",m_params.U_0Beta);
//     logFile.log("U_1",m_params.U_1);
//     logFile.log("t_0",m_params.t_0);
//     logFile.log("lambda_0",m_params.lambda_0);
//     logFile.log("t_1",m_params.t_1);
//     logFile.log("lambda_1",m_params.lambda_1);
//     logFile.log("meff",m_params.meff);
//     logFile.log("R",m_params.R);
//     logFile.log("P",m_params.P);
//     logFile.log("N",m_params.N);
//     logFile.log("M",m_params.M);
//     logFile.log("HilbertSpaceSize",m_params.HilbertSpaceSize);
//     logFile.log("}");

// }

// void SecondQuantisedHamiltonian::load(FILE *file)
// {// loads parameters from a file
//     m_loadingType = fromParameters;
//     while (true)
//     {
//         char objectName[30];
//         int ret = fscanf(file," %29[^{}:\n ] : ",objectName);
//         if (ret <= 0)
//             break;
//         std::string objectName_s(objectName);
//         if (objectName_s == "U_0Alpha")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.U_0Alpha);
//         }
//         else if (objectName_s == "U_0Beta")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.U_0Beta);
//         }
//         else if (objectName_s == "U_1")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.U_1);
//         }
//         else if (objectName_s == "t_0")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.t_0);
//         }
//         else if (objectName_s == "lambda_0")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.lambda_0);
//         }
//         else if (objectName_s == "t_1")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.t_1);
//         }
//         else if (objectName_s == "lambda_1")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.lambda_1);
//         }
//         else if (objectName_s == "meff")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.meff);
//         }
//         else if (objectName_s == "R")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.R);
//         }
//         else if (objectName_s == "P")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.P);
//         }
//         else if (objectName_s == "N")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.N);
//         }
//         else if (objectName_s == "M")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.M);
//         }
//         else if (objectName_s == "HilbertSpaceSize")
//         {
//             int dummy;
//             ConfigFileLoader::loadParameter(file,dummy);
//             //ConfigFileLoader::loadParameter(file,m_params.HilbertSpaceSize);
//         }
//         else
//         {
//             logger().log("Unknown second quantised Hamiltonian parameter: ",objectName_s);
//         }
//     }
//     m_params.HilbertSpaceSize = 2*m_params.N*m_params.M;


// }
