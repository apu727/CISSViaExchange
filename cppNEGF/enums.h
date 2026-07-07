#ifndef ENUMS_H
#define ENUMS_H
namespace enums
{
enum class IndexType
{
    direct,
    dual,
    improperDirect,
    improperDual,
    eigenValue, // For contracting eigenvalues with with each other
    invalid // Default
};
enum class SpinSymmetry
{
    NoSpin,//This matrix is not a spin matrix
    NoSpinSym, // No symmetry among the spin
    RHF,// Alpha and Beta Block are equal & no off diagonal blocks
    SpinMatrix, // This is a spin matrix, i.e. a Linear combination of Pauli matrices
    //TODO We need a spatial basis enum
};
enum class MatrixProperties
{
    None,
    //TODO AntiSelfAdjoint matrix. Useful in norm optimisations
    selfAdjoint, // The actual adjoint accounting for the Basis, satisfies adjoint(H) = H => there is a basis where it is Hermitian.
    //The normal proof that a Hermitian matrix has Orthonormal eigenvalues is:
    /*
     * Hv = lambda v
     * v^\dagger H v = lambda* v^\dagger v = lambda v^\dagger v => (lambda - lambda*) v^\dagger v = 0; => lambda = lambda*
     * Hu = lambda' u
     * u^\dagger H = lambda' u^\dagger
     * u^\dagger H v = lambda' u^\dagger v = lambda u^\dagger v => (lambda - lambda') u^\dagger v = 0;
     *
     * since the adjoint of a vector is the dual vector, (S_{\bar{j}i}c^i)^\dagger = c_j
     * And ad(Hv) = ad(v)ad(H) because ad(H^{i}_{j} c^j) = (S_{\bar{k}i}H^{i}_{j} c^j)^\dagger = (S_{\bar{k}i}H^{i}_{j}S^{j\bar{l}} S_{\bar{l}m}c^m)^\dagger = (S_{\bar{k}i}H^{i}_{j}S^{j\bar{l}})^\dagger (S_{\bar{l}m}c^m)^\dagger = ad(v)ad(H)
     * Hv = lambda v
     * ad(v) H v = lambda* ad(v) v = lambda ad(v) v => (lambda - lambda*) ad(v) v = 0; => lambda = lambda*
     * Hu = lambda' u
     * ad(u) H = lambda' ad(u)
     * ad(u) H v = lambda' ad(u) v = lambda ad(u) v => (lambda - lambda') ad(u) v = 0;
     * Note that ad(u)v = (S_{\bar{j}i}u^i)^\dagger v^j = u^{\bar{i}} S_{\bar{i}j} v^j is the normal inner product
     *
     * Therefore selfAdjoint matrices take the place of Hermitian matrices in general. The two coincide if the basis is orthonormal
     * Because we cant check if the basis is orthonormal at compile time we have to assume that Hermitian matrices are NOT! self adjoint
     */
    Hermitian, // The conjugate transpose, satisfies H^\dagger = H. Note that this does not mean it is self adjoint! unless the basis is orthonormal
    /*
     * For Hermitian, ad(H) = (S_{..} H^{.}_{.} S^{..})^\dagger =/= H in general
     * For Hermitian, ad(H) = (S^{..} H_{..} S^{..})^\dagger  = S^{..}^\dagger H_{..} S^{..}^\dagger = S^{..} H_{..} S^{..} =?= H
     *
     * Interestingly, It's eigenvalues are real. Since v^\dagger v = v^\bar{j} v^j =/=0. Even though it is an illegal contraction.
     * Further the vectors v^j are Unitary if the basis were orthonormal. This is used in eigenValue tricks but nowhere else?
     */
    Unitary, // Under conjugate transpose, satisfies U^\dagger U = I => U^\dagger = U^{-1}. Note that this does not mean it is norm preserving!
    NormPreserving, // Satisfies adjoint(U)U = I => There is a basis where it is Unitary and ad(U) = U^{-1}
};
}
#endif // ENUMS_H
