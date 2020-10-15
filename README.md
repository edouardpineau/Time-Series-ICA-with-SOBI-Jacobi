#### Implementation of the Second Order Blind Source Separation (SOBI) with Jacobi angles for joint diagonalization

SOBI is an independent componenent analysis (ICA) method for multivariate time series data that relies on the joint diagonalization of the lagged-covariance matrices of the whiten data. In this repository, we implement the efficient joint diagonalization algorithm proposed in [Jacobi angles for simultaneous diagonalization](https://www.researchgate.net/publication/277295728_Jacobi_Angles_For_Simultaneous_Diagonalization), Cardoso and al. (1996) and apply it for time series ICA.

The SOBI.py contains an implementation of the [Jacobi rotation](https://en.wikipedia.org/wiki/Jacobi_rotation). 

Useful bibliography:

[1] Cardoso, Jean-François and Souloumiac, Antoine. Jacobi Angles For Simultaneous Diagonalization. 1996.

[2] Ziehe, Andreas and Laskov, Pavel and Nolte, Guido and Müller, Klaus-Robert. A fast algorithm for joint diagonalization with non-orthogonal transformations and its application to blind source separation. 2004.
