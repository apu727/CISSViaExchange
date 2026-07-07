#ifndef GLOBAL_H
#define GLOBAL_H
#include <cmath>
#include <complex>
typedef double numType;
typedef std::complex<numType> complexType;
#define numTypeCode "%lg"

//Atomic units
const numType hbar = 1.0;
const numType e = 1.0;
const numType me = 1.0;
const numType h = 2.0 * M_PI;
const numType energyUnit = 4.3597447222060e-18; // Hartree
const numType lengthUnit = 5.29177210544e-11; // Bohr radius in metres
const numType electricPotentialUnit = 27.211386245981; // in Volts
const numType kb = 1.380649e-23 / energyUnit; //# Hartrees per kelvin
// const numType T = 6000; // Kelvin
extern numType T; // Kelvin
const numType E_0 = 0.25/M_PI; // Vaccum permitivity
const std::complex<double> iu(0.0,1.0);   // imaginary unit

extern unsigned long NUM_CORES;

void releaseAssert(bool val,const std::string& message); // defined in logger.cpp

inline const numType negInf = -std::numeric_limits<numType>::infinity();
inline const numType posInf = std::numeric_limits<numType>::infinity();

// #define DOEXTERNTEMPLATES

//Compatibility with GCC 11.4.0 and others.
//Note: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
//& https://stackoverflow.com/questions/57812038/if-constexpr-and-dependent-false-static-assert-is-ill-formed
template<class T> struct dependent_false : std::false_type {};

#define MAYBE_UNUSED __attribute__ ((unused))

#define CONST_REF_CAPTURE(x) &x = std::as_const(x)
#define BY_REF_CAPTURE(x) &x = x
#define BY_VAL_CAPTURE(x) x = x // not perfectly identitical to 'x' https://stackoverflow.com/questions/36188694/lambda-capture-to-use-the-initializer-or-not-to-use-it

#endif // GLOBAL_H
