#include "global.h"
#include "diis.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

bool simpleNewtonRaphson(const std::function<Eigen::MatrixXd (Eigen::VectorXd)> &HessianFunc,
                         const std::function<Eigen::VectorXd (Eigen::VectorXd)> &GradFunc,
                         const std::function<numType (Eigen::VectorXd)> & ErrorFunc,
                         Eigen::VectorXd& point, bool downHill)
{
    int numberOfStepsLeft = 200;
    numType zeroThreshold = 1e-11;
    numType machinePrecision = 1e-15;
    bool converged = false;
    numType err;
    Eigen::MatrixXd Hess;
    Eigen::VectorXd Grad;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    Eigen::VectorXd eVal;
    Eigen::MatrixXd eVec;

    Eigen::VectorXd step;
    Eigen::VectorXd dir;
    numType directionalDerivative;
    Eigen::VectorXd newGrad;

    while(numberOfStepsLeft-- > 0)
    {
        point.normalize();
        err = ErrorFunc(point);
        // logger().log("err",err);
        Hess = HessianFunc(point);
        Grad = GradFunc(point);
        // logger().log("GradNorm",Grad.norm());
        if (Grad.norm() < 1e-12)
        {
            converged = true;
            break;
        }
        // realNumType Error = ErrorFunc(point);

        es.compute(Hess,Eigen::ComputeEigenvectors);
        eVal = es.eigenvalues();
        eVec = es.eigenvectors();
        for (auto& e : eVal)
        {
            if (abs(e) < zeroThreshold)
                e = 0;
            else
            {
                if (downHill)
                    e = abs(1./e);
                else
                    e = 1./e;
            }
        }
        step = -(eVec * (eVal.asDiagonal() * (eVec.adjoint() * Grad)));

        if (downHill)
        {
            // numType errTest = ErrorFunc(point);
            // while (errTest > err && step.cwiseQuotient(point).lpNorm<Eigen::Infinity>() > machinePrecision)
            // {
            //     step /= 2;
            //     point -= step;
            //     errTest = ErrorFunc(point);
            // }
            //From TUPS

            constexpr size_t maxSearch = 20;
            size_t searchCount = 0;
            long double tMax = 1.5;
            long double t = 1;
            long double newT = 1;
            Eigen::VectorXd pointCopy = point;
            point += t*step;

            Eigen::VectorXd trialGradCubic;
            numType energyTrial;
            long double a;
            long double b;
            long double c;

            long double E1;
            long double E2;
            long double foundDirectionalDeriv;
            do
            {
                searchCount++;
                energyTrial = ErrorFunc(point);
                trialGradCubic = GradFunc(point);

                b = step.transpose() * (Hess * step);
                c = step.dot(Grad);

                foundDirectionalDeriv = trialGradCubic.dot(step);
                a = (foundDirectionalDeriv - (b + c))/(t*t);
                b /= t;


                long double discriminant = b*b-4*a*c;
                if (c >= 0)
                    logger().log("c >= 0", (double)c);
                if (a >= 0)
                {
                    newT = (-b + std::sqrt(discriminant))/(2*a);
                    // energyTrial = Energy + a*t*t*t/3 + b*t*t/2 + c*t;
                }
                else
                {
                    if (discriminant >= 0)
                    {
                        long double t1 = (-b + std::sqrt(discriminant))/(2*a);
                        E1 = a*t1*t1*t1/3 + b*t1*t1/2 + c*t1 + err;
                        E2 = a*tMax*tMax*tMax/3 + b*tMax*tMax/2 + c*tMax + err;
                        if (E2 < E1)
                        {
                            newT = tMax;
                            // energyTrial = E2;
                        }
                        else
                        {
                            newT = t1;
                            // energyTrial = E1;
                        }
                    }
                    else //monotone decrease up to boundary
                    {
                        newT = tMax;
                        // energyTrial = a*tMax*tMax*tMax/3 + b*tMax*tMax/2 + c*tMax + Energy;
                    }
                }
                point = pointCopy;
                point = newT*step;

                if (energyTrial > err && Grad.norm() > 1e-8)
                {
                    t = t/2;
                    tMax = tMax/2;
                    point = pointCopy;
                    point += t*step;
                    if (searchCount > maxSearch)
                        break;
                }
                else
                {
                    if (newT > tMax)
                        newT = tMax;
                    point = pointCopy;
                    point += newT*step;
                    break;
                }
            }
            while(true);
        }
        else
        {
            dir = step;
            dir.normalize();
            directionalDerivative = Grad.dot(dir);
            point += step;
            if (abs(directionalDerivative) < machinePrecision)
            {
                logger().log("Landscape too flat");
            }
            newGrad = GradFunc(point);
            while (abs(newGrad.dot(dir)) > abs(directionalDerivative) && step.cwiseQuotient(point).lpNorm<Eigen::Infinity>() > machinePrecision)
            {
                step /= 2;
                point -= step;
                newGrad = GradFunc(point);
            }
        }

        if (!(step.lpNorm<Eigen::Infinity>() > machinePrecision))
        {
            fprintf(stderr, "Backtracking failed in simpleNewtonRaphson\n");
            break;
        }
    }
    int saddleIndex = 0;
    es.compute(Hess,Eigen::ComputeEigenvectors);
    eVal = es.eigenvalues();
    eVec = es.eigenvectors();
    for (auto& e : eVal)
    {
        if (e < -zeroThreshold)
            saddleIndex++;
    }
    logger().log("saddleIndex",saddleIndex);
    logger().log("GradNorm",Grad.norm());
    logger().log("numberOfStepsLeft",numberOfStepsLeft);
    logger().log("converged",converged);
    return converged;
}
