
cdef extern from "gelnet.h":
  void gelnet_logreg_opt( double* X, int* y, double* d, double* K, double* m,
      double* lambda1p, double* lambda2p,
      double* S, double* Kwm, int* np, int* pp,
      int* max_iter, double* eps, double* w, double* b,
      int* bSilentp, int* bBalanced, int* bNonneg );
      
  void gelnet_lin_opt( double* X, double* z, double* a, double* d, double* K, 
        double* m, double* lambda1p, double* lambda2p,
        double* S, double* Kwm, int* np, int* pp,
        int* max_iter, double* eps, int* fix_bias,
        double* w, double* b, int* bSilentp, int* bNonneg );