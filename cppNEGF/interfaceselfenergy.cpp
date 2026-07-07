#include "interfaceselfenergy.h"

template<typename SelfEnergyMatrixType>
complexType InterfaceSelfEnergy<SelfEnergyMatrixType>::getKFromE(complexType E)
{
    if (E.imag() == 0)
    {
        numType RE = E.real();
        if (m_params.m_type == LeadType::TightBindingInfinite)
        {
            if (RE <= m_params.U0 + 2*m_params.t)
                return 0;
            else if (RE == m_params.U0 - 2*m_params.t)
                return M_PI;
            else if (RE < m_params.U0 - 2*m_params.t)
            {// E = U0 + 2*t*cos(ka)
                return std::acos((RE-m_params.U0)/(2*m_params.t))/m_params.a;
            }
            else
                return 0;
        }
        if (m_params.m_type == LeadType::ContinuumInfinite)
        {
            if (RE-m_params.U0 < 0)
                return -1; // valid k is always positive so this is fine.
            else
                return std::sqrt((RE-m_params.U0)*2*m_params.m);
        }
        else
            releaseAssert(false, "not implemented");
    }
    else
    {
        //Guarantee not on the branch cut
        if (m_params.m_type == LeadType::TightBindingInfinite)
        {

            // E = U0 + 2*t*cos(ka)
            return std::acos((E-m_params.U0)/(2*m_params.t))/m_params.a;
        }
        if (m_params.m_type == LeadType::ContinuumInfinite)
        {
            return std::sqrt((E-m_params.U0)*2.*m_params.m);
        }
        else
            releaseAssert(false, "not implemented");
    }
    //never reached
    releaseAssert(false, "unreachable");
    return 0;

}

template<class SelfEnergyMatrixType>
complexType InterfaceSelfEnergy<SelfEnergyMatrixType>::fermiFunction(complexType E)
{
    if (E.real()-m_params.mu > 40*kb*T)
        return 0;
    else
        return 1.0/(exp((E-m_params.mu)/(kb*T)) + 1.);

}

template<typename SelfEnergyMatrixType>
InterfaceSelfEnergy<SelfEnergyMatrixType>::InterfaceSelfEnergy(Parameters params)
{
    m_params = params;
    if (m_params.m_type == LeadType::TightBindingInfinite || m_params.m_type == LeadType::ContinuumInfinite)
    {
        ComplexMatrix couplePoints;
        m_params.couplingPoints.toBasisC(couplePoints,m_params.couplingPoints.getBasis());
        couplePoints.setSpinSym(enums::SpinSymmetry::RHF);

        ComplexMatrix magnetisedCouplePoints;
        BasisManager::possibleTags tag = BasisManager::possibleTags::NullBasis;
        const auto & bas = m_params.couplingPoints.getBasis();
        auto noSpinBasis_i = BasisManager::removeSpinTags(bas[0],&tag);
        auto noSpinBasis_j = BasisManager::removeSpinTags(bas[1],&tag);

        releaseAssert(tag != BasisManager::possibleTags::NullBasis,"No spin tag");
        magnetisedCouplePoints.setZero(2*couplePoints.rows(),2*couplePoints.cols(),{noSpinBasis_i,noSpinBasis_j},enums::SpinSymmetry::NoSpinSym);
        auto spinMatrix =(1-m_params.MagnetisationDirection.norm())*pauliMatrices::sigmaI + m_params.MagnetisationDirection[0] * pauliMatrices::sigmaX +  m_params.MagnetisationDirection[1] * pauliMatrices::sigmaY + m_params.MagnetisationDirection[2] * pauliMatrices::sigmaZ;

        for (long i = 0; i < couplePoints.rows(); i++)
        {
            for (long j = 0; j < couplePoints.cols(); j++)
            {
                magnetisedCouplePoints(2*i,2*j) = spinMatrix(0,0)*couplePoints(i,j);
                magnetisedCouplePoints(2*i,2*j+1) = spinMatrix(0,1)*couplePoints(i,j);
                magnetisedCouplePoints(2*i+1,2*j) = spinMatrix(1,0)*couplePoints(i,j);
                magnetisedCouplePoints(2*i+1,2*j+1) = spinMatrix(1,1)*couplePoints(i,j);
            }
        }
        m_SigmaRBase = (magnetisedCouplePoints * magnetisedCouplePoints.adjoint()).sparseView();
        if (m_params.m_type == LeadType::ContinuumInfinite)
            m_params.U0 += 2*m_params.t;
        this->m_type = selfEnergyType::Analytic;
    }
    else if (m_params.m_type == LeadType::GreensFunction)
    {
        releaseAssert((bool)m_params.Gfunc,"Gfunc not set");
        releaseAssert(m_params.SECouple.size() > 0,"SECouple not set");
        for (auto& s : m_params.SECouple)
        {
            releaseAssert(s->m_type == selfEnergyType::ConstantSelfConsistent || s->m_type == selfEnergyType::Constant,"Self energy must be constant or constant self consistent");
            releaseAssert(s->hasSigmaK() == false,"Self Energy must not have a SigmaK");
        }
        m_currentBasis = {m_params.Gfunc->getBasis(),m_params.Gfunc->getBasis()};
        this->m_type = selfEnergyType::Analytic;

    }
    else
        releaseAssert(false, "not implemented");
}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::getSigmaIn(numType E, SelfEnergyMatrixType &dest)
{
    getGamma(E,dest);
    dest *= fermiFunction(E);
}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::getSigmaOut(numType E, SelfEnergyMatrixType &dest)
{
    getGamma(E,dest);
    dest *= (1.-fermiFunction(E));
}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::getSigmaR(complexType E, SelfEnergyMatrixType &dest)
{
    if (m_params.m_type == LeadType::TightBindingInfinite || m_params.m_type == LeadType::ContinuumInfinite)
    {
        complexType k = getKFromE(E);
        complexType Gr;
        if (m_params.m_type == LeadType::TightBindingInfinite)
            Gr = std::exp(iu*k*m_params.a)/m_params.t;
        else if (m_params.m_type == LeadType::ContinuumInfinite)
        {
            if (k.imag() == 0)
            {
                numType Rk = k.real();
                if (Rk > 0)
                    Gr = -2*m_params.m*std::exp(iu*Rk*m_params.a)*std::sin(Rk*m_params.a)/Rk;
                else if (Rk == 0)
                    Gr = -2*m_params.m*m_params.a;
                else
                    Gr = 0;
            }
            else
            {
                Gr = -2*m_params.m*std::exp(iu*k*m_params.a)*std::sin(k*m_params.a)/k;
            }
        }
        dest = m_SigmaRBase*Gr;
    }
    else if (m_params.m_type == LeadType::GreensFunction)
    {
        ComplexMatrix Gr;
        m_params.Gfunc->getGrAtE(E,Gr);

        if constexpr (SelfEnergyMatrixType::isSparse)
        {
            dest = (m_params.SECoupleCacheLHS * Gr * m_params.SECoupleCacheRHS).sparseView();
        }
        else
        {
            dest = m_params.SECoupleCacheLHS * Gr * m_params.SECoupleCacheRHS;
        }
    }
    else
        releaseAssert(false, "not implemented");
}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::getSigmaA(complexType E, SelfEnergyMatrixType &dest)
{
    if (m_params.m_type == LeadType::GreensFunction)
    {
        ComplexMatrix Ga;
        m_params.Gfunc->getGaAtE(E,Ga);

        if constexpr (SelfEnergyMatrixType::isSparse)
        {
            dest = (m_params.SECoupleCacheAdjointLHS * Ga * m_params.SECoupleCacheAdjointRHS).sparseView();
        }
        else
        {
            dest = m_params.SECoupleCacheAdjointLHS * Ga * m_params.SECoupleCacheAdjointRHS;
        }
    }
    else
    {
        SelfEnergyMatrixType temp;
        getSigmaR(std::conj(E),temp);
        dest = temp.adjoint();
    }
}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::getSigmaK(numType E, SelfEnergyMatrixType &dest)
{
    getGamma(E,dest);
    dest *= -iu*(1.-2.*fermiFunction(E));
}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::getGamma(numType E, SelfEnergyMatrixType &dest)
{
    SelfEnergyMatrixType SA;
    getSigmaR(E,dest);
    // SA = dest.adjoint();
    getSigmaA(E,SA);
    dest -= SA;
    dest *= iu;
}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::recomputeSelfEnergies()
{
    if(m_params.m_type == LeadType::GreensFunction)
    {
        auto GRbas = m_params.Gfunc->getBasis();
        SelfEnergyMatrixType temp,temp2;
        for (size_t i = 0; i < m_params.SECouple.size(); i++)
        {

            if (i == 0)
            {
                m_params.SECouple[i]->getSigmaR(temp);
                temp.toBasisC(m_params.SECoupleCacheRHS,{GRbas,m_currentBasis[1]});
                temp.toBasisC(m_params.SECoupleCacheLHS,{m_currentBasis[0],GRbas});

                m_params.SECouple[i]->getSigmaA(temp);
                temp.toBasisC(m_params.SECoupleCacheAdjointRHS,{GRbas,m_currentBasis[1]});
                temp.toBasisC(m_params.SECoupleCacheAdjointLHS,{m_currentBasis[0],GRbas});
            }
            else
            {
                m_params.SECouple[i]->getSigmaR(temp);
                temp.toBasisC(temp2,{GRbas,m_currentBasis[1]});
                m_params.SECoupleCacheRHS += temp2;

                temp.toBasisC(temp2,{m_currentBasis[0],GRbas});
                m_params.SECoupleCacheLHS += temp2;

                m_params.SECouple[i]->getSigmaA(temp);
                temp.toBasisC(temp2,{GRbas,m_currentBasis[1]});
                m_params.SECoupleCacheAdjointRHS += temp2;
                temp.toBasisC(temp2,{m_currentBasis[0],GRbas});
                m_params.SECoupleCacheAdjointLHS += temp2;
            }
        }
    }
}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::setMatrixDimension(uint32_t)
{

}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::clearCache()
{

}

template<typename SelfEnergyMatrixType>
void InterfaceSelfEnergy<SelfEnergyMatrixType>::setBasis(const std::string &bas)
{
    if (bas != m_currentBasis[0])
    {
        if (m_params.m_type == LeadType::TightBindingInfinite || m_params.m_type== LeadType::ContinuumInfinite)
        {
            ComplexSparseMatrix newSigmaRBase;
            m_SigmaRBase.toBasisC(newSigmaRBase,bas);
            m_SigmaRBase = newSigmaRBase;
            m_currentBasis = {bas,bas};
        }
        else if(m_params.m_type == LeadType::GreensFunction)
        {
            //Handled in recomputeSelfEnergies
            m_currentBasis = {bas,bas};
            recomputeSelfEnergies();
        }
    }
    recomputeSelfEnergies();
}

template<typename SelfEnergyMatrixType>
std::vector<numType> InterfaceSelfEnergy<SelfEnergyMatrixType>::getContinuousRegions()
{
    if(m_params.m_type == LeadType::GreensFunction && false)
    {
        std::vector<numType> ret;
        numType start = -10;
        numType end = 10;
        size_t steps = 1000;
        numType step = (end-start)/steps;
        ret.reserve(steps+3);
        ret.push_back(negInf);
        for (size_t i = 0; i < steps+1; i++)
        {
            ret.push_back(start + i*step);
        }
        ret.push_back(posInf);
        return ret;
    }
    else
        return {negInf,posInf};
}

template class InterfaceSelfEnergy<ComplexMatrix>;
template class InterfaceSelfEnergy<ComplexSparseMatrix>;
