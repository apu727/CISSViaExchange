#include "logger.h"
#include "linalg.h"
#include <iostream>

bool tosilence = false;

logger::logger()
{
    if (tosilence)
        m_file = fopen("/dev/null","w");
    else
        m_file = stderr;
}

void logger::silence()
{
    tosilence = true;
}

void logger::unsilence()
{
    tosilence = false;
}

logger::logger(std::string filename, bool append)
{
    if (append)
        m_file = fopen(filename.c_str(), "a");
    else
        m_file = fopen(filename.c_str(), "w");
    if (m_file == nullptr)
    {
        logger().log("Could not open file for logging",filename);
        logger().log("Error:",errno);
        m_file = stderr;
    }
    if (append)
        log("Start Logging\n");
}

logger::~logger()
{
    if (m_file != stderr)
        fclose(m_file);
}

void logger::log(const std::vector<std::string> &names, const std::vector<numType> &quantities)
{
    for (int i = 0; i < names.size(); i++)
    {
        fprintf(m_file, "%s: " numTypeCode ", " , names[i].c_str(), quantities[i]);
    }
    fprintf(m_file,"\n");
}

void logger::log(const std::vector<std::string> &names, const std::vector<complexType> &quantities)
{
    for (int i = 0; i < names.size(); i++)
    {
        fprintf(m_file, "%s: " numTypeCode " + " numTypeCode"j, " , names[i].c_str(), quantities[i].real(),quantities[i].imag());
    }
    fprintf(m_file,"\n");
}

void logger::log(std::string name, numType quantity){fprintf(m_file, "%s: " numTypeCode "\n", name.c_str(), quantity);}

void logger::log(std::string name, complexType quantity){fprintf(m_file, "%s: " numTypeCode "," numTypeCode "\n", name.c_str(), quantity.real(),quantity.imag());}

void logger::log(std::string name, uint32_t quantity){fprintf(m_file, "%s: %u\n", name.c_str(), quantity);}

void logger::log(const std::string& message)
{
    fprintf(m_file, "%s\n", message.c_str());

}


void logger::log(std::string name, std::string value)
{
    fprintf(m_file, "%s: %s\n", name.c_str(), value.c_str());
}

void logger::log(std::string name, const Eigen::Matrix<complexType, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
    std::cerr << name << std::endl << mat << std::endl;
}

void logger::logAccurate(std::string name, const std::string &quantity){log(name,quantity);}

void logger::logAccurate(std::string name, numType quantity){fprintf(m_file, "%s: %.16lf\n", name.c_str(), quantity);}

void logger::logAccurate(std::string name, complexType quantity){fprintf(m_file, "%s: %.16lf,%.16lf\n", name.c_str(), quantity.real(),quantity.imag());}
template<>
void logger::log(std::string name, const std::vector<numType> &object)
{
    fprintf(m_file, "%s: [", name.c_str());
    for (auto& o :object)
        fprintf(m_file, numTypeCode ", ", o);
    fprintf(m_file, "]\n");
}

void releaseAssert(bool val, const std::string& message)
{
    if (!val)
    {
        logger().log("Release Assert Failed: ",message);
        __builtin_trap();
    }
}


void logger::log(std::string name, int quantity){fprintf(m_file, "%s: %i\n", name.c_str(), quantity);}

void logger::log(std::string name, long quantity){fprintf(m_file, "%s: %li\n", name.c_str(), quantity);}

void logger::log(std::string name, size_t quantity){fprintf(m_file, "%s: %zu\n", name.c_str(), quantity);}

void logger::log(std::string name, std::atomic<int> &quantity) {log(name, (int)quantity);}




void logger::logAccurate(std::string name, const Eigen::Matrix<numType, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
    fprintf(m_file,"%s: {",name.c_str());
    for (long i = 0; i < mat.rows();i++)
    {
        for (long j = 0; j < mat.cols(); j++)
        {
            if (i == mat.rows()-1 && j == mat.cols()-1)
                fprintf(m_file,"%.16lf",mat(i,j));
            else
                fprintf(m_file,"%.16lf,",mat(i,j));
        }
    }
    fprintf(m_file,"}\n");
}

void logger::logAccurate(std::string name, const Eigen::Matrix<complexType, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
    fprintf(m_file,"%s: {",name.c_str());
    for (long i = 0; i < mat.rows();i++)
    {
        for (long j = 0; j < mat.cols(); j++)
        {
            if (i == mat.rows()-1 && j == mat.cols()-1)
                fprintf(m_file,"%.16lf,%.16lf",mat(i,j).real(),mat(i,j).imag());
            else
                fprintf(m_file,"%.16lf,%.16lf,",mat(i,j).real(),mat(i,j).imag());
        }
    }
    fprintf(m_file,"}\n");
}

void logger::logAccurate(std::string name, const Eigen::SparseMatrix<numType> &mat)
{
    logAccurate(name,mat.toDense());
}

void logger::logAccurate(std::string name, const Eigen::SparseMatrix<complexType> &mat)
{
    logAccurate(name,mat.toDense());
}

void logger::logAccurate(std::string name, const Eigen::Matrix<complexType, Eigen::Dynamic, 1> &mat)
{
    logAccurate(name,static_cast<const Eigen::Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic>&>(mat));
}

void logger::logAccurate(std::string name, const Eigen::Matrix<numType, Eigen::Dynamic, 1> &mat)
{
    logAccurate(name,static_cast<const Eigen::Matrix<numType,Eigen::Dynamic,Eigen::Dynamic>&>(mat));
}

//Implementation of forward declarations in logger.h, Note that all template specialisations must forward declared otherwise weird linker bugs
template<>
void logger::log(std::string name, const std::vector<std::complex<double>>& object)
{
    fprintf(m_file, "%s: [", name.c_str());
    for (const std::complex<double>& o :object)
        fprintf(m_file, "%s:%s, ", std::to_string(o.real()).c_str(), std::to_string(o.imag()).c_str());
    fprintf(m_file, "]\n");
}

template<>
void logger::logAccurate(std::string name, const std::vector<numType>& object)
{
    fprintf(m_file, "%s: [", name.c_str());
    for (const numType& o :object)
        fprintf(m_file, "%.16lf, ", o);
    fprintf(m_file, "]\n");
}
template<>
void logger::logAccurate(std::string name, const std::vector<complexType>& object)
{
    fprintf(m_file, "%s: [", name.c_str());
    for (const complexType& o :object)
        fprintf(m_file, "(%.16lf,%.16lf), ", o.real(),o.imag());
    fprintf(m_file, "]\n");
}


//Enums
template <>
void logger::logAccurate<enums::IndexType>(const std::string& name, enums::IndexType e)
{
    switch (e)
    {
    case enums::IndexType::direct:
        log(name,"direct"); break;
    case enums::IndexType::dual:
        log(name,"dual"); break;
    case enums::IndexType::improperDirect:
        log(name,"improperDirect"); break;
    case enums::IndexType::improperDual:
        log(name,"improperDual"); break;
    case enums::IndexType::eigenValue:
        log(name,"eigenValue"); break;
    case enums::IndexType::invalid:
        log(name,"invalid"); break;
    default:
        log(name,static_cast<std::underlying_type_t<enums::IndexType>>(e)); break;

    };
}



