#include "linalg.h"


#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/native_enum.h>
#include <pybind11/operators.h>

// XXMultiplication means that XX has an XX type index on the left
#define directMultiplication ,ComplexDirectVector// no such types exist yet

#define dualMultiplication ,RealMatrix,RealHermitianMatrix,ComplexMatrix,ComplexHermitianMatrix,ComplexSelfAdjointMatrix,\
ComplexDualMatrix,ComplexDualHermitianMatrix,ComplexDualHermitianMatrix,ComplexDualSelfAdjointMatrix,\
    ComplexSparseMatrix,ComplexHermitianSparseMatrix,ComplexSelfAdjointSparseMatrix,\
    ComplexDualSparseMatrix,ComplexDualHermitianSparseMatrix,ComplexDualSelfAdjointSparseMatrix,\
    ComplexDualVector, EigenVector

#define improperDirectMultiplication ,ComplexDirectMatrix,ComplexDirectHermitianMatrix,ComplexDirectSelfAdjointMatrix,\
        ComplexDirectSparseMatrix,ComplexDirectHermitianSparseMatrix,ComplexDirectSelfAdjointSparseMatrix
#define improperDualMultiplication    //No such types exist yet

#define eigenValueMultiplication , RealEigenValueMatrix, EigenValueMatrix

#define AllMatrices directMultiplication  dualMultiplication improperDirectMultiplication improperDualMultiplication eigenValueMultiplication

namespace py = pybind11;

template<typename T, int sizei, int sizej, enums::MatrixProperties Params, enums::IndexType m_iIndexT, enums::IndexType m_jIndexT>
typename Matrix<T, sizei, sizej, Params, m_iIndexT, m_jIndexT>::MatType Matrix<T, sizei, sizej, Params, m_iIndexT, m_jIndexT>::pyGetMatrixBuffer() const
{
    return m_MatBuffer.get();
}

template <typename BaseMat>
void declareMulOperators(py::class_<BaseMat>& c){}

//This is horrible!
template <typename BaseMat, typename other, typename ... others>
void declareMulOperators(py::class_<BaseMat>& c)
{
    if constexpr (BaseMat::static_jIndexT == indexContractionPair<other::static_iIndexT>::v && // Matrices cannot be contracted
                  BaseMat::static_sizej == other::static_sizei )// Matrices wrong size for contraction - Matrix Multiplication
    {
        c.def(py::self * other());
    }

    if constexpr (BaseMat::static_iIndexT == other::static_iIndexT && BaseMat::static_jIndexT == other::static_jIndexT &&
                  BaseMat::static_sizei == other::static_sizei && BaseMat::static_sizej == other::static_sizej &&
                  std::is_same_v<typename BaseMat::static_T,typename other::static_T> )
    {
        c.def(py::self + other());
        c.def(py::self - other());
    }

    if constexpr (MatrixAdditionParams<BaseMat::static_Params,other::static_Params>::v == BaseMat::static_Params &&
                  BaseMat::static_iIndexT == other::static_iIndexT && BaseMat::static_jIndexT == other::static_jIndexT &&
                  BaseMat::static_sizei == other::static_sizei && BaseMat::static_sizej == other::static_sizej &&
                  !BaseMat::isSparse && std::is_same_v<typename BaseMat::static_T,typename other::static_T>)
    {
        c.def(py::self += other());
        c.def(py::self -= other());
    }
    if constexpr (BaseMat::isSparse == other::isSparse &&
                  std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T> &&// Basis change Scalar type is wrong. Likely needs to be complex
                  (other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None))// Basis change Parameters are incorrect
    {
        c.def("toBasis",
              [](const BaseMat& self, const other&, const std::string& bas, bool silenceSpinBlockWarning)
              {
                  other o;
                  bool s = self.toBasis(o,bas,silenceSpinBlockWarning);
                  if (s)
                      return o;
                  else
                      throw std::invalid_argument(("Cannot convert to basis: " + bas));
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
        c.def("toBasis",
              [](const BaseMat& self, const other&, const indexBasisT& bas, bool silenceSpinBlockWarning)
              {
                  other o;
                  bool s = self.toBasis(o,bas,silenceSpinBlockWarning);
                  if (s)
                      return o;
                  else
                      throw std::invalid_argument(("Cannot convert to basis: " + bas[0] + "and" + bas[1]));
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
        c.def("toBasisCast",
              [](const BaseMat& self, const other&, const std::string& bas, bool silenceSpinBlockWarning)
              {
                  constexpr bool isSparse = BaseMat::isSparse;

                  std::conditional_t<isSparse,typename changeMatrixParameters<other, enums::MatrixProperties::None>::Sparse,typename changeMatrixParameters<other, enums::MatrixProperties::None>::Dense> o;
                  bool s = self.toBasis(o,bas,silenceSpinBlockWarning);

                  if (s)
                  {
                      if constexpr (std::is_same_v<other,decltype(o)>)
                          return o;
                      else
                          return static_cast<other>(o);
                  }
                  else
                      throw std::invalid_argument(("Cannot convert to basis: " + bas));
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
        c.def("toBasisCast",
              [](const BaseMat& self, const other&, const indexBasisT& bas, bool silenceSpinBlockWarning)
              {
                  constexpr bool isSparse = BaseMat::isSparse;

                  std::conditional_t<isSparse,typename changeMatrixParameters<other, enums::MatrixProperties::None>::Sparse,typename changeMatrixParameters<other, enums::MatrixProperties::None>::Dense> o;
                  bool s = self.toBasis(o,bas,silenceSpinBlockWarning);

                  if (s)
                  {
                      if constexpr (std::is_same_v<other,decltype(o)>)
                          return o;
                      else
                          return static_cast<other>(o);
                  }
                  else
                      throw std::invalid_argument(("Cannot convert to basis: " + bas[0] + " and " + bas[1]));
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
    }
    else if constexpr (BaseMat::isSparse == other::isSparse &&
               std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>)
    {
        c.def("toBasis",
              [](const BaseMat& self, const other&, const std::string& bas, bool silenceSpinBlockWarning)
              {
                  logger().log("Static asserts failed for requested basis transform type");
                  logger().log("BaseMat::isSparse == other::isSparse",BaseMat::isSparse == other::isSparse);
                  logger().log("std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>",std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>);
                  logger().log("(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None))",(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None));
                  throw std::invalid_argument(("Cannot convert to basis: " + bas + "Static assert checks failed. Try toBasisCast"));
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
        c.def("toBasis",
              [](const BaseMat& self, const other&, const indexBasisT& bas, bool silenceSpinBlockWarning)
              {
                  logger().log("Static asserts failed for requested basis transform type");
                  logger().log("BaseMat::isSparse == other::isSparse",BaseMat::isSparse == other::isSparse);
                  logger().log("std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>",std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>);
                  logger().log("(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None))",(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None));
                  throw std::invalid_argument(("Cannot convert to basis: " + bas[0] + " and " + bas[1] + "Static assert checks failed. Try toBasisCast"));
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
        c.def("toBasisCast",
              [](const BaseMat& self, const other&, const std::string& bas, bool silenceSpinBlockWarning)
              {
                  constexpr bool isSparse = BaseMat::isSparse;

                  std::conditional_t<isSparse,typename changeMatrixParameters<other, enums::MatrixProperties::None>::Sparse,typename changeMatrixParameters<other, enums::MatrixProperties::None>::Dense> o;
                  bool s = self.toBasis(o,bas,silenceSpinBlockWarning);

                  if (s)
                  {
                      if constexpr (std::is_same_v<other,decltype(o)>)
                        return o;
                      else
                          return static_cast<other>(o);
                  }
                  else
                      throw std::invalid_argument(("Cannot convert to basis: " + bas));
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
        c.def("toBasisCast",
              [](const BaseMat& self, const other&, const indexBasisT& bas, bool silenceSpinBlockWarning)
              {
                  constexpr bool isSparse = BaseMat::isSparse;

                  std::conditional_t<isSparse,typename changeMatrixParameters<other, enums::MatrixProperties::None>::Sparse,typename changeMatrixParameters<other, enums::MatrixProperties::None>::Dense> o;
                  bool s = self.toBasis(o,bas,silenceSpinBlockWarning);

                  if (s)
                  {
                      if constexpr (std::is_same_v<other,decltype(o)>)
                          return o;
                      else
                          return static_cast<other>(o);
                  }
                  else
                      throw std::invalid_argument(("Cannot convert to basis: " + bas[0] + " and " + bas[1]));
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
    }
    else
    {
        c.def("toBasis",
              [](const BaseMat& self, const other&, const std::string& bas, bool silenceSpinBlockWarning)
              {
                  logger().log("Static asserts failed for requested basis transform type");
                  logger().log("BaseMat::isSparse == other::isSparse",BaseMat::isSparse == other::isSparse);
                  logger().log("std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>",std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>);
                  logger().log("(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None))",(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None));
                  throw std::invalid_argument("Static asserts failed for requested basis transform type");
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
        c.def("toBasis",
              [](const BaseMat& self, const other&, const indexBasisT& bas, bool silenceSpinBlockWarning)
              {
                  logger().log("Static asserts failed for requested basis transform type");
                  logger().log("BaseMat::isSparse == other::isSparse",BaseMat::isSparse == other::isSparse);
                  logger().log("std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>",std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>);
                  logger().log("(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None))",(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None));
                  throw std::invalid_argument("Static asserts failed for requested basis transform type");
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);
        c.def("toBasisCast",
              [](const BaseMat& self, const other&, const indexBasisT& bas, bool silenceSpinBlockWarning)
              {
                  logger().log("Static asserts failed for requested basis transform type");
                  logger().log("BaseMat::isSparse == other::isSparse",BaseMat::isSparse == other::isSparse);
                  logger().log("std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>",std::is_same_v<std::complex<typename BaseMat::realT>,typename other::static_T>);
                  logger().log("(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None))",(other::static_Params == (BaseMat::static_Params == enums::MatrixProperties::selfAdjoint ? enums::MatrixProperties::selfAdjoint : enums::MatrixProperties::None) || other::static_Params == enums::MatrixProperties::None));
                  throw std::invalid_argument("Static asserts failed for requested basis transform type even with casts");
              },
              py::arg("Desired Matrix Type"),py::arg("desiredBasis"), py::arg("silenceSpinBlockWarning") = false);

    }
    if constexpr (other::static_iIndexT == BaseMat::static_iIndexT && other::static_jIndexT == BaseMat::static_jIndexT &&
                  other::static_sizei == BaseMat::static_sizei && other::static_sizej == BaseMat::static_sizej &&
                  (std::is_same_v<typename other::static_T, typename BaseMat::static_T> || std::is_same_v<typename other::static_T, typename BaseMat::complexT>) && !other::isSparse)
        c.def("to",[](const BaseMat& self, const other& ot){return static_cast<other>(self);});
    //This works but lots of ram.
    //Static asserts to pass are:
    //            static_assert(checkCompileTimeMatrixMultiplication<complexT,sizej,sizei,Params,indexContractionPair<m_jIndexT>::v,indexContractionPair<m_iIndexT>::v
    //                                                           ,T2,sizei2,sizej2,Params2,m_iIndexT2,m_jIndexT2>::v , "Matrix Multiplication Check Failed ");
    //static_assert(checkCompileTimeTrace<complexT,sizej,sizej2,MatrixMultiplicationParams<Params,Params2>::v,indexContractionPair<m_jIndexT>::v,m_jIndexT2>::v , "Matrix Trace Check Failed");
    // if constexpr(BaseMat::isSparse == other::isSparse &&
    //             BaseMat::static_iIndexT == other::static_iIndexT && BaseMat::static_sizei == other::static_sizei && //mat mul
    //             BaseMat::static_jIndexT == other::static_jIndexT && BaseMat::static_sizej == other::static_sizej) //trace
    //     c.def("cdot",[](const BaseMat& self, const other& oth){return self.cdot(oth);},py::arg("Other"));

    declareMulOperators<BaseMat,others...>(c);
}




//Linalg Prep
template <typename T, typename ... mulOthers>
void declare_Matrix(py::module &m, const std::string& niceName)
{
    typedef typename T::static_T scalar;
    constexpr int sizei = T::static_sizei ;
    constexpr int sizej = T::static_sizej;
    constexpr enums::MatrixProperties Params = T::static_Params;
    constexpr enums::IndexType m_iIndexT = T::static_iIndexT;
    constexpr enums::IndexType m_jIndexT = T::static_jIndexT;

    auto c = py::class_<T>(m, niceName.c_str())
                 .def(py::init<>())
                 .def(py::init<long, long, const std::string&, enums::SpinSymmetry>(),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def(py::init<long, long, const indexBasisT&, enums::SpinSymmetry>(),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)

                 .def(py::init<const typename T::MatType&, const std::string&, enums::SpinSymmetry>(),
                      py::arg("init"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def(py::init<const typename T::MatType&, const indexBasisT&, enums::SpinSymmetry>(),
                      py::arg("init"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def("setZero", py::overload_cast<>(&T::setZero))
                 .def("setRandom", &T::setRandom)

                 .def("setZero", py::overload_cast<long, long, const std::string&, enums::SpinSymmetry>(&T::setZero),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)                 
                 .def("setZero", py::overload_cast<long, long, const indexBasisT&, enums::SpinSymmetry>(&T::setZero),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)

                 .def("setIdentity", py::overload_cast<long, long, const std::string&, enums::SpinSymmetry>(&T::setIdentity),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def("setIdentity", py::overload_cast<long, long, const indexBasisT&, enums::SpinSymmetry>(&T::setIdentity),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)

                 .def("resize", py::overload_cast<long, long, const std::string&, enums::SpinSymmetry>(&T::resize),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def("resize", py::overload_cast<long, long, const indexBasisT&, enums::SpinSymmetry>(&T::resize),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 //This is a templated function so need to define all possible basis changes
                 // .def("toBasis",&T::toBasisC,
                 //               py::arg("dest"), py::arg("desiredBasis"))

                 // .def("adjoint", &T::adjoint) BROKEN
                 .def("conjugateTranspose", &T::conjugateTranspose)
                 .def("inverse",&T::inverse)
                 .def("rows", &T::rows)
                 .def("cols", &T::cols)
                 .def("getCol", &T::getCol, py::arg("i"))
                 .def("getRow", &T::getRow, py::arg("i"))
                 // .def("coeffRef", &T::coeffRef, py::arg("i"), py::arg("j"))
                 // .def("coeff", &T::coeff, py::arg("i"), py::arg("j"))
                 //.def("Operator(i,j) non const)
                 .def("sparseView", &T::sparseView)
                 .def("getSpinSym", &T::getSpinSym)
                 .def("setSpinSym", &T::setSpinSym, py::arg("sym"))
                 .def_static("Identity", py::overload_cast<long,long,const std::string&,enums::SpinSymmetry>(&T::Identity),
                             py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def_static("Zero", py::overload_cast<long,long,const std::string&,enums::SpinSymmetry>(&T::Zero),
                             py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def("getBasis",&T::getBasis)
                 .def("getBasisS",&T::getBasisS)
                 .def("setBasis",py::overload_cast<const std::string&>(&T::setBasis))
                 .def("setBasis",py::overload_cast<const indexBasisT&>(&T::setBasis))
                 .def("toNumpy",[](const T& self){return self.pyGetMatrixBuffer();});


    //Not defining the *= and /= operators for complex because there are too many cases. multiplication by reals is always allowed
    c.def(py::self * typename T::realT());
    c.def(typename T::realT() * py::self);
    c.def(py::self *= typename T::realT());

    c.def(py::self /  typename T::realT());
    c.def(py::self /=  typename T::realT());

    if constexpr (!T::isReal)
    {
        c.def(py::self * typename T::complexT());
        c.def(typename T::complexT() * py::self);

        c.def(py::self /  typename T::complexT());
    }

    declareMulOperators<T,mulOthers...>(c);
    if constexpr(T::static_sizei == 1)
    {
        c.def("__getitem__",
              [](const T& self, const py::tuple& index)
              {
                  if (py::len(index) > 2 || py::len(index) == 0)
                  {
                      throw py::cast_error("Invalid size, expected 2 or 1");
                  }

                  if (py::len(index) == 2 && index[0].cast<long>() != 1)
                      throw py::index_error("Index i out of range");

                  long i = 1;
                  long j;

                  if (py::len(index) == 2)
                      j = index[1].cast<long>() != 1;
                  else
                      j = index[0].cast<long>() != 1;

                  if (j < 0  || j > self.rows())
                      throw py::index_error("Index i out of range");
                  return self.coeff(i,j);
              },py::arg("index"));
        c.def("__setitem__",
              [](T& self, const py::tuple& index,typename T::static_T& value)
              {
                  if (py::len(index) > 2 || py::len(index) == 0)
                  {
                      throw py::cast_error("Invalid size, expected 2 or 1");
                  }

                  if (py::len(index) == 2 && index[0].cast<long>() != 1)
                      throw py::index_error("Index i out of range");

                  long i = 1;
                  long j;

                  if (py::len(index) == 2)
                      j = index[1].cast<long>() != 1;
                  else
                      j = index[0].cast<long>() != 1;

                  if (j < 0  || j > self.rows())
                      throw py::index_error("Index i out of range");
                  self.coeffRef(i,j) = value;
        },py::arg("index"), py::arg("value"));
    }
    else if constexpr(T::static_sizej == 1)
    {
        c.def("__getitem__",
              [](const T& self, const py::tuple& index)
              {
                  if (py::len(index) > 2 || py::len(index) == 0)
                  {
                      throw py::cast_error("Invalid size, expected 2 or 1");
                  }
                  if (py::len(index) == 2 && index[1].cast<long>() != 1)
                      throw py::index_error("Index j out of range");

                  long i = index[0].cast<long>();
                  long j = 1;
                  if (i < 0  || i > self.rows())
                      throw py::index_error("Index i out of range");
                  return self.coeff(i,j);
            },py::arg("index"));
        c.def("__setitem__",
              [](T& self, const py::tuple& index,typename T::static_T& value)
              {
                  if (py::len(index) > 2 || py::len(index) == 0)
                  {
                      throw py::cast_error("Invalid size, expected 2 or 1");
                  }
                  if (py::len(index) == 2 && index[1].cast<long>() != 1)
                      throw py::index_error("Index j out of range");

                  long i = index[0].cast<long>();
                  long j = 1;
                  if (i < 0  || i > self.rows())
                      throw py::index_error("Index i out of range");
                  self.coeffRef(i,j) = value;
              },py::arg("index"), py::arg("value"));
    }
    else
    {
        c.def("__getitem__",
              [](const T& self, const py::tuple& index)
              {
                  if (py::len(index) != 2)
                  {
                      throw py::cast_error("Invalid size, expected 2");
                  }
                  long i = index[0].cast<long>();
                  long j = index[1].cast<long>();
                  if (i < 0  || i > self.rows())
                      throw py::index_error("Index i out of range");
                  if (j < 0  || j > self.cols())
                      throw py::index_error("Index j out of range");
                  return self.coeff(i,j);
              },py::arg("index"));
        c.def("__setitem__",
              [](T& self, const py::tuple& index,typename T::static_T& value)
              {
                  if (py::len(index) != 2)
                  {
                      throw py::cast_error("Invalid size, expected 2");
                  }
                  long i = index[0].cast<long>();
                  long j = index[1].cast<long>();
                  if (i < 0  || i > self.rows())
                      throw py::index_error("Index i out of range");
                  if (j < 0  || j > self.cols())
                      throw py::index_error("Index j out of range");
                  self.coeffRef(i,j) = value;
              },py::arg("index"), py::arg("value"));
    }




    if constexpr(m_iIndexT == indexContractionPair<m_jIndexT>::v && sizei == sizej)
        c.def("trace", &T::trace);
    if constexpr(m_iIndexT == indexContractionPair<m_jIndexT>::v && sizei == sizej)
    {
        c.def("getEigenValues",
              [](const T& self)
              {
                  typename T::EigenValueType eVals;
                  bool s = self.getEigenValues(eVals);
                  if (!s) logger().log("Failed getting eigenvalues");
                  return eVals;
              }
              );
        c.def("getEigenValuesAndVectors",
              [](const T& self, bool diagonalConditioner)
              {
                  typename T::EigenValueType eVals;
                  typename T::EigenVectorType eVecs;
                  bool s = self.getEigenValuesAndVectors(eVals,eVecs,diagonalConditioner);
                  if (!s) logger().log("Failed getting eigenvalues + Vectors");
                  return py::make_tuple(eVals,eVecs);
              },py::arg("diagonalConditioner") = false);
    }
    c.def("norm",&T::norm);
    c.def("squaredNorm",&T::squaredNorm);
    if constexpr(m_iIndexT == enums::IndexType::dual && m_jIndexT == enums::IndexType::direct)
    {
        c.def("asBasisChangeMatrix", &T::asBasisChangeMatrix);
        c.def("asProjectionMatrix",&T::asProjectionMatrix);
    }
    if constexpr(m_iIndexT == enums::IndexType::improperDirect && m_jIndexT == enums::IndexType::direct && Params == enums::MatrixProperties::Hermitian)
        c.def("asMetric", &T::asMetric);
    //These dont compile: "Holder classes are only supported for custom types" include/pybind11/cast.h line 766 As of 4/08/2025
    // if constexpr(m_iIndexT == enums::IndexType::dual && m_jIndexT == enums::IndexType::direct)
    //     c.def("asTemporaryBasisChangeMatrix", &T::asTemporaryBasisChangeMatrix);
    // if constexpr(m_iIndexT == enums::IndexType::improperDirect && m_jIndexT == enums::IndexType::direct && Params == enums::MatrixProperties::Hermitian)
    //     c.def("asTemporaryMetric", &T::asTemporaryMetric);
    if constexpr(sizei == 1)
    {
        c.def(py::init<long, const std::string&, enums::SpinSymmetry>(),
              py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin);
    }
    else if constexpr (sizej == 1)
    {
        c.def(py::init<long, const std::string&, enums::SpinSymmetry>(),
              py::arg("rows"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin);
    }
}

template <typename T, typename ... mulOthers>
void declare_Sparse(py::module &m, const std::string& niceName)
{
    typedef typename T::static_T scalar;
    constexpr int sizei = T::static_sizei ;
    constexpr int sizej = T::static_sizej;
    constexpr enums::MatrixProperties Params = T::static_Params;
    constexpr enums::IndexType m_iIndexT = T::static_iIndexT;
    constexpr enums::IndexType m_jIndexT = T::static_jIndexT;

    auto c = py::class_<T>(m, niceName.c_str())
                 .def(py::init<>())
                 .def(py::init<long, long, const std::string&, enums::SpinSymmetry>(),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def(py::init<const typename T::MatType&, const std::string&, enums::SpinSymmetry>(),
                      py::arg("init"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def("setZero", py::overload_cast<>(&T::setZero))
                 .def("setZero", py::overload_cast<long, long, const std::string&, enums::SpinSymmetry>(&T::setZero),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def("resize", py::overload_cast<long, long, const std::string&, enums::SpinSymmetry>(&T::resize),
                      py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 //This is a templated function so need to define all possible basis changes
                 // .def("toBasis",&T::toBasisC,
                 //               py::arg("dest"), py::arg("desiredBasis"))

                 // .def("adjoint", &T::adjoint) BROKEN
                 .def("conjugateTranspose", &T::conjugateTranspose)
                 .def("rows", &T::rows)
                 .def("cols", &T::cols)
                 .def("coeffRef", &T::coeffRef, py::arg("i"), py::arg("j"))
                 .def("coeff", &T::coeff, py::arg("i"), py::arg("j"))
                 //.def("Operator(i,j) non const)
                 .def("toDense", [](const T& self) {
                     return static_cast<Matrix<typename T::static_T, Eigen::Dynamic, Eigen::Dynamic, T::static_Params, T::static_iIndexT, T::static_jIndexT>>(self);
                 })
                 .def("getSpinSym", &T::getSpinSym)
                 .def("makeCompressed", &T::makeCompressed)
                 .def_static("Zero", py::overload_cast<long, long, const std::string&, enums::SpinSymmetry>(&T::Zero),
                             py::arg("rows"), py::arg("cols"), py::arg("basis"), py::arg("sym") = enums::SpinSymmetry::NoSpin)
                 .def("getBasis",&T::getBasis)
        ;

    c.def("norm",&T::norm);
    declareMulOperators<T,mulOthers...>(c);
}

extern template void declare_Matrix<RealMatrix AllMatrices>(py::module &,const std::string&);

extern template void declare_Matrix<RealHermitianMatrix AllMatrices>(py::module &,const std::string&);

extern template void declare_Matrix<ComplexMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<ComplexHermitianMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<ComplexSelfAdjointMatrix AllMatrices>(py::module &,const std::string&);

extern template void declare_Matrix<ComplexDirectMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<ComplexDirectHermitianMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<ComplexDirectSelfAdjointMatrix AllMatrices>(py::module &,const std::string&);

extern template void declare_Matrix<ComplexDualMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<ComplexDualHermitianMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<ComplexDualSelfAdjointMatrix AllMatrices>(py::module &,const std::string&);

extern template void declare_Sparse<ComplexSparseMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Sparse<ComplexHermitianSparseMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Sparse<ComplexSelfAdjointSparseMatrix AllMatrices>(py::module &,const std::string&);

extern template void declare_Sparse<ComplexDirectSparseMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Sparse<ComplexDirectHermitianSparseMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Sparse<ComplexDirectSelfAdjointSparseMatrix AllMatrices>(py::module &,const std::string&);

extern template void declare_Sparse<ComplexDualSparseMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Sparse<ComplexDualHermitianSparseMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Sparse<ComplexDualSelfAdjointSparseMatrix AllMatrices>(py::module &,const std::string&);


// declare_Matrix<RaisingMetric>(m,"RaisingMetric");
// declare_Matrix<LoweringMetric>(m,"LoweringMetric");

// declare_Matrix<ForwardTransformMatrix>(m,"ForwardTransformMatrix");
// declare_Matrix<InverseTransformMatrix>(m,"InverseTransformMatrix");

//Spin matrices generally
extern template void declare_Matrix<HermitianMatrix2cd>(py::module &,const std::string&);
extern template void declare_Matrix<UnitaryMatrix2cd>(py::module &,const std::string&);

//Vectors
extern template void declare_Matrix<ComplexDualVector AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<ComplexDirectVector AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<EigenValueVector AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<RealEigenValueVector>(py::module &,const std::string&);

extern template void declare_Matrix<EigenValueMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<RealEigenValueMatrix AllMatrices>(py::module &,const std::string&);

extern template void declare_Matrix<EigenVectorMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<InvEigenVectorMatrix AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<UnitaryEigenVectorMatrix AllMatrices>(py::module &,const std::string&);

extern template void declare_Matrix<EigenVector AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<InvEigenVector AllMatrices>(py::module &,const std::string&);

extern template void declare_Matrix<UnitaryEigenVector AllMatrices>(py::module &,const std::string&);
extern template void declare_Matrix<UnitaryInvEigenVector AllMatrices>(py::module &,const std::string&);

void declare_linalg(py::module &m); //in PythonLinalg2.cpp

// declare_Matrix<RealMatrix dualMultiplication>(m,"RealMatrix");
// declare_Matrix<RealHermitianMatrix dualMultiplication>(m,"RealHermitianMatrix");

// declare_Matrix<ComplexMatrix dualMultiplication>(m,"ComplexMatrix");
// declare_Matrix<ComplexHermitianMatrix dualMultiplication>(m,"ComplexHermitianMatrix");
// declare_Matrix<ComplexSelfAdjointMatrix dualMultiplication>(m,"ComplexSelfAdjointMatrix");

// declare_Matrix<ComplexDirectMatrix dualMultiplication>(m,"ComplexDirectMatrix");
// declare_Matrix<ComplexDirectHermitianMatrix dualMultiplication>(m,"ComplexDirectHermitianMatrix");
// declare_Matrix<ComplexDirectSelfAdjointMatrix dualMultiplication>(m,"ComplexDirectSelfAdjointMatrix");

// declare_Matrix<ComplexDualMatrix improperDirectMultiplication>(m,"ComplexDualMatrix");
// declare_Matrix<ComplexDualHermitianMatrix improperDirectMultiplication>(m,"ComplexDualHermitianMatrix");
// declare_Matrix<ComplexDualSelfAdjointMatrix improperDirectMultiplication>(m,"ComplexDualSelfAdjointMatrix");

// declare_Sparse<ComplexSparseMatrix dualMultiplication>(m,"ComplexSparseMatrix");
// declare_Sparse<ComplexHermitianSparseMatrix dualMultiplication>(m,"ComplexHermitianSparseMatrix");
// declare_Sparse<ComplexSelfAdjointSparseMatrix dualMultiplication>(m,"ComplexSelfAdjointSparseMatrix");

// declare_Sparse<ComplexDirectSparseMatrix dualMultiplication>(m,"ComplexDirectSparseMatrix");
// declare_Sparse<ComplexDirectHermitianSparseMatrix dualMultiplication>(m,"ComplexDirectHermitianSparseMatrix");
// declare_Sparse<ComplexDirectSelfAdjointSparseMatrix dualMultiplication>(m,"ComplexDirectSelfAdjointSparseMatrix");

// declare_Sparse<ComplexDualSparseMatrix improperDirectMultiplication>(m,"ComplexDualSparseMatrix");
// declare_Sparse<ComplexDualHermitianSparseMatrix improperDirectMultiplication>(m,"ComplexDualHermitianSparseMatrix");
// declare_Sparse<ComplexDualSelfAdjointSparseMatrix improperDirectMultiplication>(m,"ComplexDualSelfAdjointSparseMatrix");

// //Aliases for cleaner code
// m.attr("RaisingMetric") = m.attr("ComplexDualHermitianMatrix");
// m.attr("LoweringMetric") = m.attr("ComplexDirectHermitianMatrix");
// m.attr("ForwardTransformMatrix") = m.attr("ComplexMatrix");
// m.attr("InverseTransformMatrix") = m.attr("ComplexMatrix");

// // declare_Matrix<RaisingMetric>(m,"RaisingMetric");
// // declare_Matrix<LoweringMetric>(m,"LoweringMetric");

// // declare_Matrix<ForwardTransformMatrix>(m,"ForwardTransformMatrix");
// // declare_Matrix<InverseTransformMatrix>(m,"InverseTransformMatrix");

// //Spin matrices generally
// declare_Matrix<HermitianMatrix2cd>(m,"HermitianMatrix2cd");
// declare_Matrix<UnitaryMatrix2cd>(m,"UnitaryMatrix2cd");

// //Vectors
// declare_Matrix<ComplexDualVector>(m,"ComplexDualVector");
// declare_Matrix<ComplexDirectVector dualMultiplication>(m,"ComplexDirectVector");
// declare_Matrix<EigenValueVector>(m,"EigenValueVector");
// declare_Matrix<RealEigenValueVector>(m,"RealEigenValueVector");

// declare_Matrix<EigenValueMatrix eigenValueMultiplication>(m,"EigenValueMatrix");
// declare_Matrix<RealEigenValueMatrix eigenValueMultiplication>(m,"RealEigenValueMatrix");

// declare_Matrix<EigenVectorMatrix eigenValueMultiplication>(m,"EigenVectorMatrix");
// declare_Matrix<InvEigenVectorMatrix dualMultiplication>(m,"InvEigenVectorMatrix");
// declare_Matrix<UnitaryEigenVectorMatrix eigenValueMultiplication>(m,"UnitaryEigenVectorMatrix");

// declare_Matrix<EigenVector>(m,"EigenVector");
// declare_Matrix<InvEigenVector dualMultiplication>(m,"InvEigenVector");

// declare_Matrix<UnitaryEigenVector>(m,"UnitaryEigenVector");
// declare_Matrix<UnitaryInvEigenVector eigenValueMultiplication>(m,"UnitaryInvEigenVector");
