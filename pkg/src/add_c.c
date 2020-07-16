#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

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

static R_CMethodDef c_methods[] = {
  {"_add_c", (DL_FUNC) &add_c, 4, NULL},
  {NULL,     NULL,             0, NULL}
};

static const R_CallMethodDef call_methods[] = {
  {"_add_call", (DL_FUNC) &add_call, 2},
  {NULL,        NULL,                0}
};

void R_init_pkg(DllInfo *info) {
  R_registerRoutines(info, c_methods, call_methods, NULL, NULL);
  R_useDynamicSymbols(info, FALSE);
  // R_forceSymbols(info, TRUE); // disable for now
}
