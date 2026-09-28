#ifndef DIIS_H
#define DIIS_H
#include "logger.h"
#include "Eigen/Dense"
#include <list>
#include <utility>

bool simpleNewtonRaphson(const std::function<Eigen::MatrixXd (Eigen::VectorXd)> &HessianFunc,
                         const std::function<Eigen::VectorXd (Eigen::VectorXd)> &GradFunc,
                         const std::function<numType (Eigen::VectorXd)> &ErrorFunc,
                         Eigen::VectorXd& point, bool downhill = false);

template <typename errorVectorType, typename quantityType, auto innerProductFunc, bool EDIIS>
class DIIS
{
    size_t m_maxDIISSize;
    std::list<errorVectorType> m_pastEVecs;
    std::list<quantityType> m_pastQuantities;
public:
    DIIS(size_t maxDIISSize)
    {
        m_maxDIISSize = maxDIISSize;
    }
    void reset()
    {
        m_pastEVecs.clear();
        m_pastQuantities.clear();
    }
    void addNew(const quantityType& quantity, const errorVectorType& eVec)
    {
        m_pastEVecs.push_back(eVec);
        m_pastQuantities.push_back(quantity);
        if (m_pastEVecs.size() > m_maxDIISSize)
        {
            m_pastEVecs.pop_front();
            m_pastQuantities.pop_front();
        }
    }

    void getNext(quantityType& nextQuantity, errorVectorType& extrapolatedError, Eigen::VectorXd* CoeffVecOut = nullptr)
    {
        releaseAssert(m_pastEVecs.size() != 0,"DIIS must have at least one vector in it");

        size_t currDIISSize = m_pastEVecs.size();

        Eigen::MatrixXd B(currDIISSize,currDIISSize);


        auto iIterator = m_pastEVecs.begin();

        for (size_t i = 0; i < currDIISSize; i++,iIterator++)
        {
            auto jIterator = m_pastEVecs.begin();
            for(size_t j = 0; j < currDIISSize; j++,jIterator++)
            {
                B(i,j) = innerProductFunc(*iIterator,*jIterator);
            }
        }
        // Eigen::Matrix<numType,CDIISSize,CDIISSize> BInv(B.inverse());
        Eigen::VectorXd rhsVec(currDIISSize);
        rhsVec.setConstant(1);
        Eigen::VectorXd CoeffVec;

        if constexpr(EDIIS)
        {
            //Normally
            //Error = c_i B_{ij} c_j - \lambda( c_ii -1)
            //Derivative_i = B_{ij} c_j + c_j B_{ji}  - \lambda
            //For Real inner products, usually the case, things B_{ij} = B_{ji}
            //Derivative_i = 2B_{ij} c_j - \lambda
            //This leads to Pulay DIIS

            //Substituting c_i = t_i^2 gives
            //Error = t^2_i B_{ij} t^2_j - \lambda( t_i t_i -1)
            //Derivative_i = (t_i 4B_{ij} t^2_j - 2\lambda t_i No sum on i
            //Derivative_lambda = 1-t_i t_i
            //Hessian_{ij} = \delta_{ij} 4B_{jk} t^2_k + 8 t_i B_{ij} t_j - 2\lambda \delta_{ij} No sum on i or j
            //Hessian_{iLambda} = -2t_i
            //Hessian_{Lambda i} = -2t_i
            //Hessian_lambdaLambda = 0

            //Without a constraint we can actually go downhill
            // E = \frac{t^2_i B_{ij} t^2_j}{(t_l t_l)^2}
            // Deriv_i = 4\frac{t_i B_{ij} t^2_j}{(t_l t_l)^2} -4E \frac{t_i}{t_l t_l}   No sum on i
            // Hessian_{ij} = \delta_{ij}\left(4\frac{ B_{ik} t^2_k}{(t_l t_l)^2} -4E \frac{1}{t_l t_l}\right) + 8\frac{t_i B_{ij} t_j}{(t_l t_l)^2} \\
            //                 &+ \left(4\frac{t_j B_{ik} t^2_k}{(t_l t_l)^2} -4E \frac{t_j}{t_l t_l}\right)\left(-4\frac{t_i}{t_l t_l}\right)\\
            //                 &  -8E \frac{t_i t_j}{(t_l t_l)^2} + \left(4\frac{t_i B_{ik} t^2_k}{(t_l t_l)^2} -4E \frac{t_i}{t_l t_l}\right) \left(-4 \frac{t_j}{t_l t_l}\right)\\
            // Hessian_{ij} &= \delta_{ij}\left(4\frac{ B_{ik} t^2_k}{(t_l t_l)^2} -4E \frac{1}{t_l t_l}\right) + 8\frac{t_i B_{ij} t_j}{(t_l t_l)^2}\\
            //                 &+ Deriv_i \left(-4 \frac{t_j}{t_l t_l}\right) + Deriv_j\left(-4 \frac{t_i}{t_l t_l}\right)\\
            //                 &- 8E \frac{t_i t_j}{(t_l t_l)^2}
            B /= B.norm();
            auto ErrorFunc = [CONST_REF_CAPTURE(B)](const Eigen::VectorXd& point)
            {
                Eigen::VectorXd tSq = point.cwiseProduct(point);
                numType normSq = tSq.sum();
                numType error = tSq.adjoint() * B * tSq;
                error /= normSq*normSq;
                return error;
            };

            auto GradFunc = [CONST_REF_CAPTURE(B),CONST_REF_CAPTURE(ErrorFunc)](const Eigen::VectorXd& point)
            {
                const Eigen::VectorXd& t = point;
                Eigen::VectorXd tSq = point.cwiseProduct(point);
                double normSq = t.squaredNorm();

                Eigen::VectorXd GradTs = 4*(B * tSq).cwiseProduct(t)/(normSq*normSq); // 4\frac{t_i B_{ij} t^2_j}{(t_l t_l)^2}
                numType error = ErrorFunc(point);
                GradTs -= error * 4 * t /normSq; //-4E \frac{t_i}{t_l t_l}

                return GradTs;
            };



            auto HessianFunc = [CONST_REF_CAPTURE(B),CONST_REF_CAPTURE(ErrorFunc),CONST_REF_CAPTURE(GradFunc)](const Eigen::VectorXd& point)
            {
                const Eigen::VectorXd& t = point;
                Eigen::VectorXd tSq = point.cwiseProduct(point);
                double normSq = t.squaredNorm();
                Eigen::VectorXd deriv = GradFunc(point);
                numType error = ErrorFunc(point);

                Eigen::MatrixXd Hess(point.rows(),point.rows());
                Hess = (4*B*tSq/(normSq*normSq)).asDiagonal(); //\delta_{ij}*4*\frac{ B_{ik} t^2_k}{(t_l t_l)^2}
                for (long i = 0; i < point.rows(); i++)
                {
                    for (long j = 0; j < point.rows(); j++)
                    {
                        Hess(i,j) += 8*t[i] *B(i,j)*t[j]/(normSq*normSq); //8\frac{t_i B_{ij} t_j}{(t_l t_l)^2}
                    }
                    Hess(i,i) -= error * 4/normSq; //-4 Error \frac{1}{t_l t_l}\delta_{ij}
                }

                Hess +=  -4/(normSq)* deriv * t.adjoint(); //Derivative_i \left(-4 \frac{t_j}{t_l t_l}\right)
                Hess +=  -4/(normSq)* t * deriv.adjoint(); //Derivative_j \left(-4 \frac{t_i}{t_l t_l}\right)
                Hess +=  -8*error/(normSq*normSq)* t * t.adjoint(); //- 12 Error \frac{t_i t_j}{(t_l t_l)^2}

                return Hess;
            };

            Eigen::VectorXd point(currDIISSize);
            point.setRandom();
            point.normalize();

            simpleNewtonRaphson(HessianFunc,GradFunc,ErrorFunc,point,true);
            CoeffVec = point.cwiseProduct(point)/(point.squaredNorm());
            Eigen::VectorXd logVector = CoeffVec.array().log();
            double entropy = -CoeffVec.cwiseProduct(logVector).sum();
            if (entropy < 1e-10)
            {
                logger().log("Forward progress mitigation, entropy",entropy);
                CoeffVec.setZero();
                CoeffVec(currDIISSize-1) = 1;
            }
            else
            {
                logger().logAccurate("CoeffVec Normalisation", CoeffVec.sum());
                logger().log("CoeffVecEntropy", entropy);
                logger().log("MaxEntropy", log(currDIISSize) );
            }
            

        }
        else
        {
            CoeffVec = Eigen::VectorXd(B.completeOrthogonalDecomposition().solve(rhsVec));
            // Preconditioned version see arXiv:2112.08890v1 eq 14.
            numType totalSum = CoeffVec.sum();

            if (totalSum == 0)
            {
                CoeffVec.setZero();
                CoeffVec(CoeffVec.rows()-1) = 1;
            }
            else
                CoeffVec /= totalSum; // makes sure the coeffs add to 1.
        }

        auto pastquantityIt = m_pastQuantities.begin();
        auto errorIt = m_pastEVecs.begin();


        for (size_t i = 0; i < currDIISSize; )
        {
            if (i == 0)
            { // avoid the problem with initialisation
                nextQuantity = CoeffVec[i] * *pastquantityIt;
                extrapolatedError = CoeffVec[i] * *errorIt;
            }
            else
            {
                nextQuantity += CoeffVec[i] * *pastquantityIt;
                extrapolatedError += CoeffVec[i] * *errorIt;
            }

            i++;
            pastquantityIt++;
            errorIt++;
        }
        //Calc error

        logger().log("DIIS Extrapolated error norm",std::sqrt(innerProductFunc(extrapolatedError,extrapolatedError)));
        logger().log("DIIS Curr error norm",std::sqrt(innerProductFunc(m_pastEVecs.back(),m_pastEVecs.back())));
        if (CoeffVecOut)
            *CoeffVecOut = std::move(CoeffVec);
    }


};

template <typename errorVectorType, typename quantityType, auto innerProductFunc>
using PulayDIIS = DIIS<errorVectorType,quantityType,innerProductFunc,false>;


template <typename errorVectorType, typename quantityType, auto innerProductFunc>
using EDIIS = DIIS<errorVectorType,quantityType,innerProductFunc,true>;


#endif // DIIS_H
