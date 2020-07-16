#include <R.h>
#include <Rinternals.h>

void add_c(const double *a, const double *b, const int *n, double *value) {
  for (int i = 0; i < *n; ++i) {
    value[i] = a[i] + b[i];
  }
}

SEXP add_call(SEXP a, SEXP b) {
  int n = length(a);
  SEXP ret = PROTECT(allocVector(REALSXP, n));
  add_c(REAL(a), REAL(b), &n, REAL(ret));
  UNPROTECT(1);
  return ret;
}
