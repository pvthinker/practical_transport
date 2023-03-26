from numba import njit

eps = 1e-12
g1 = 0.1
g2 = 0.6
g3 = 0.3
k1 = 13./12.
k2 = 0.25

f1 = 1./3.
f2 = 2./3.


@njit(fastmath=True)
def weno3(qm, q0, qp):
    """
    Third order WENO reconstruction
    """
    qi1 = -0.5*qm + 1.5*q0
    qi2 = 0.5*(q0 + qp)

    beta1 = (q0-qm)**2
    beta2 = (qp-q0)**2

    w1 = f1 / (beta1+eps)**2
    w2 = f2 / (beta2+eps)**2

    return (w1*qi1 + w2*qi2) / (w1 + w2)


@njit(fastmath=True)
def weno5(qmm, qm, q0, qp, qpp):
    """
    Fifth-order WENO reconstruction, from:
        Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
        Journal of Computation Physics 126, 202–228 (1996)
    """
    qi1 = 1./3.*qmm - 7./6.*qm + 11./6.*q0
    qi2 = -1./6.*qm + 5./6.*q0 + 1./3.*qp
    qi3 = 1./3.*q0 + 5./6.*qp - 1./6.*qpp

    beta1 = k1 * (qmm-2*qm+q0)**2 + k2 * (qmm-4*qm+3*q0)**2
    beta2 = k1 * (qm-2*q0+qp)**2 + k2 * (qm-qp)**2
    beta3 = k1 * (q0-2*qp+qpp)**2 + k2 * (3*q0-4*qp+qpp)**2

    w1 = g1 / (beta1+eps)**2
    w2 = g2 / (beta2+eps)**2
    w3 = g3 / (beta3+eps)**2

    return (w1*qi1 + w2*qi2 + w3*qi3) / (w1 + w2 + w3)
