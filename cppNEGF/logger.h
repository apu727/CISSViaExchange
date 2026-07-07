#ifndef LOGGER_H
#define LOGGER_H

#include "global.h"
#include "enums.h"
//Note that we cannot include linalg.h here since it depends on logger
#include "linalgForwardDeclarations.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class logger
{
    FILE* m_file = nullptr;
public:
    static void silence();
    static void unsilence();

    logger(std::string filename, bool append = false);
    logger();
    virtual ~logger();
    void log(const std::string& message);
    void log(const std::vector<std::string>& names, const std::vector<numType>& quantities);
    void log(const std::vector<std::string>& names, const std::vector<complexType>& quantities);
    void log(std::string name, numType quantity);
    void log(std::string name, complexType quantity);
    void log(std::string name, uint32_t quantity);
    void log(std::string name, int quantity);
    void log(std::string name, long quantity);
    void log(std::string name, size_t quantity);
    void log(std::string name, std::atomic<int>& quantity);
    void log(std::string name, std::string value);
    template<typename T>
    void log(std::string name, const std::vector<T>& object)
    {
        fprintf(m_file, "%s: [", name.c_str());
        for (const T& o :object)
            fprintf(m_file, "%s, ", std::to_string(o).c_str());
        fprintf(m_file, "]\n");
    }

    void log(std::string name, const Eigen::Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic>& mat);

    template<typename T>
    void log(std::string name, T quantity, const std::string& format_code)
    {
        fprintf(m_file, (std::string("%s: ") + format_code + std::string("\n")).c_str(), name.c_str(), quantity);
    }

    template<int8_t arrSize>
    void log(std::string name, int (&arr)[arrSize])
    {
        fprintf(m_file, "%s: [", name.c_str());
        for (size_t i = 0; i < arrSize; i++)
            fprintf(m_file, "%i, ", arr[i]);
        fprintf(m_file, "]\n");
    }

    template<typename T>
    void logAccurate(std::string name, const std::vector<T>& object)
    {
        fprintf(m_file, "%s: [", name.c_str());
        for (const T& o :object)
             fprintf(m_file, "%s, ", std::to_string(o).c_str());
        fprintf(m_file, "]\n");
    }

    template<typename T>
    void logAccurate(std::string name, const std::vector<std::shared_ptr<T>>& object)
    {
        fprintf(m_file, "%s: [", name.c_str());
        for (const std::shared_ptr<T>& o :object)
            fprintf(m_file, "%p, ", o.get());
        fprintf(m_file, "]\n");
    }

    // Add a specialisation to this for all the enum types you wish to print. Make sure this specialisation is visible whenever the enum type is visible to avoid ill-formed program NDR
    // template <>
    // inline void logger::logAccurate<MYENUMTYPE>(const std::string& name, MYENUMTYPE e)
    // {
    //     switch (e)
    //     {
    //     default:
    //         log(name,static_cast<std::underlying_type_t<MYENUMTYPE>>(e)); break;

    //     };
    // }
    template <typename E, typename std::enable_if_t<std::is_enum_v<E>,bool> = 1>
    void logAccurate(const std::string& name,E e);

    template <typename E, typename std::enable_if_t<std::is_enum_v<E>,bool> = 1>
    void log(const std::string& name,E e){logAccurate(name,e);};

    void logAccurate(std::string name, const std::string& quantity);
    void logAccurate(std::string name, numType quantity);
    void logAccurate(std::string name, complexType quantity);
    void logAccurate(std::string name, const Eigen::Matrix<numType,Eigen::Dynamic,Eigen::Dynamic>& mat);
    void logAccurate(std::string name, const Eigen::Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic>& mat);
    //Utility
    void logAccurate(std::string name, const Eigen::Matrix<numType,Eigen::Dynamic,1>& mat);

    void logAccurate(std::string name, const Eigen::SparseMatrix<numType>& mat);
    void logAccurate(std::string name, const Eigen::SparseMatrix<complexType>& mat);
    void logAccurate(std::string name, const Eigen::Matrix<complexType,Eigen::Dynamic,1>& mat);
    template <typename T1, int sizei1, int sizej1, enums::MatrixProperties Params1,
             enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
    void logAccurate(std::string name, const Matrix<T1,sizei1,sizej1,Params1,m_iIndexT1,m_jIndexT1>& mat);

    template <typename T1, enums::MatrixProperties Params1,
             enums::IndexType m_iIndexT1, enums::IndexType m_jIndexT1>
    void logAccurate(std::string name, const SparseMatrix<T1,Params1,m_iIndexT1,m_jIndexT1>& mat);


    template <typename ...args>
    void printf(const char* str, args&& ... arg)
    {
        fprintf(m_file,str,std::forward<args>(arg)...);
    }

};

//Explicit specialisations in .cpp forward declaration
template<> void logger::log(std::string name, const std::vector<std::complex<double>>& object);
template<> void logger::logAccurate(std::string name, const std::vector<numType>& object);
template<> void logger::logAccurate(std::string name, const std::vector<complexType>& object);

//Implement the templates here as the enums header would have circular dependencies
template <> void logger::logAccurate<enums::IndexType>(const std::string& name, enums::IndexType e);




#endif // LOGGER_H
