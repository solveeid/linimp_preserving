# linimp_preserving
Local and global energy-preserving methods for the Korteweg-de Vries (KdV) equation and the Zakharov-Kuznetsov (ZK) equation.

Recommended packages/tools in Matlab: None. Recommended packages in Python: numpy, time, scipy.sparse, matplotlib.pyplot.

This code is meant as a supplement to [1], and is an implementation of the linearly implicit methods preserving a discrete approximation to local and global conservation laws for the KdV or Zakharov-Kuznetsov equations. Please refer to [1] if this code is used in a project. Also implemented are the fully implicit local and global energy preserving schemes of [2] and (in the Matlab code for the KdV equation) the implicit midpoint and multi-symplectic box schemes of [3].

The Python code is less efficient than the Matlab code. This is because the solving of the linear systems is not optimized.

This is the first release, so if you have comments and suggestions for improvements, please contact me and I will try to accommodate them.

[1] Eidnes, S., and Li, L. "Linearly implicit local and global energy-preserving methods for Hamiltonian PDEs." arXiv preprint arXiv:1907.02122 (2019).

[2] Gong, Y., Cai, J., and Wang, Y. "Some new structure-preserving algorithms for general multi-symplectic formulations of Hamiltonian PDEs." Journal of Computational Physics 279 (2014): 80-102.

[3] Ascher, U. M., and McLachlan, R. I. (2005). "On symplectic and multisymplectic schemes for the KdV equation." Journal of Scientific Computing 25.1 (2005): 83-104.
