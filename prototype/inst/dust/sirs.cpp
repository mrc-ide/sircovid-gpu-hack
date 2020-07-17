// Generated by odin.dust (version 0.0.5) - do not edit
class sirs {
public:
  typedef int int_t;
  typedef double real_t;
  struct init_t {
    real_t alpha;
    real_t beta;
    real_t gamma;
    real_t I_ini;
    real_t initial_I;
    real_t initial_R;
    real_t initial_S;
    real_t p_IR;
    real_t p_RS;
    real_t S_ini;
  };
  sirs(const init_t& data): internal(data) {
  }
  size_t size() {
    return 3;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(3);
    state[0] = internal.initial_S;
    state[1] = internal.initial_I;
    state[2] = internal.initial_R;
    return state;
  }
  void update(size_t step, const std::vector<real_t>& state, dust::RNG<real_t, int_t>& rng, std::vector<real_t>& state_next) {
    const real_t S = state[0];
    const real_t I = state[1];
    const real_t R = state[2];
    real_t N = S + I + R;
    real_t n_IR = rng.rbinom(std::round(I), internal.p_IR);
    real_t n_RS = rng.rbinom(std::round(R), internal.p_RS);
    real_t p_SI = 1 - std::exp(- internal.beta * I / (real_t) N);
    real_t n_SI = rng.rbinom(std::round(S), p_SI);
    state_next[2] = R + n_IR - n_RS;
    state_next[1] = I + n_SI - n_IR;
    state_next[0] = S - n_SI + n_RS;
  }
private:
  init_t internal;
};
#include <array>
#include <cpp11/R.hpp>
#include <cpp11/sexp.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/strings.hpp>
#include <vector>

// These would be nice to make constexpr but the way that NA values
// are defined in R's include files do not allow it.
template <typename T>
inline T na_value();

template <>
inline int na_value<int>() {
  return NA_INTEGER;
}

template <>
inline double na_value<double>() {
  return NA_REAL;
}

template <typename T>
inline bool is_na(T x);

template <>
inline bool is_na(int x) {
  return x == NA_INTEGER;
}

template <>
inline bool is_na(double x) {
  return ISNA(x);
}

inline size_t object_length(cpp11::sexp x) {
  return ::Rf_xlength(x);
}

template <typename T>
void user_check_value(T value, const char *name, T min, T max) {
  if (ISNA(value)) {
    cpp11::stop("'%s' must not be NA", name);
  }
  if (min != na_value<T>() && value < min) {
    cpp11::stop("Expected '%s' to be at least %g", name, (double) min);
  }
  if (max != na_value<T>() && value > max) {
    cpp11::stop("Expected '%s' to be at most %g", name, (double) max);
  }
}

template <typename T>
void user_check_array_value(const std::vector<T>& value, const char *name,
                            T min, T max) {
  for (auto& x : value) {
    user_check_value(x, name, min, max);
  }
}

inline size_t user_get_array_rank(cpp11::sexp x) {
  if (!::Rf_isArray(x)) {
    return 1;
  } else {
    cpp11::integers dim = cpp11::as_cpp<cpp11::integers>(x.attr("dim"));
    return dim.size();
  }
}

template <size_t N>
void user_check_array_rank(cpp11::sexp x, const char *name) {
  size_t rank = user_get_array_rank(x);
  if (rank != N) {
    if (N == 1) {
      cpp11::stop("Expected a vector for '%s'", name);
    } else if (N == 2) {
      cpp11::stop("Expected a matrix for '%s'", name);
    } else {
      cpp11::stop("Expected an array of rank %d for '%s'", N, name);
    }
  }
}

template <size_t N>
void user_check_array_dim(cpp11::sexp x, const char *name,
                          const std::array<int, N>& dim_expected) {
  cpp11::integers dim = cpp11::as_cpp<cpp11::integers>(x.attr("dim"));
  for (size_t i = 0; i < N; ++i) {
    if (dim[(int)i] != dim_expected[i]) {
      Rf_error("Incorrect size of dimension %d of '%s' (expected %d)",
               i + 1, name, dim_expected[i]);
    }
  }
}

template <>
inline void user_check_array_dim<1>(cpp11::sexp x, const char *name,
                                    const std::array<int, 1>& dim_expected) {
  if ((int)object_length(x) != dim_expected[0]) {
    cpp11::stop("Expected length %d value for '%s'", dim_expected[0], name);
  }
}

template <size_t N>
void user_set_array_dim(cpp11::sexp x, const char *name,
                        std::array<int, N>& dim) {
  cpp11::integers dim_given = cpp11::as_cpp<cpp11::integers>(x.attr("dim"));
  std::copy(dim_given.begin(), dim_given.end(), dim.begin());
}

template <>
inline void user_set_array_dim<1>(cpp11::sexp x, const char *name,
                                  std::array<int, 1>& dim) {
  dim[0] = object_length(x);
}

template <typename T>
T user_get_scalar(cpp11::list user, const char *name,
                  const T previous, T min, T max) {
  T ret = previous;
  cpp11::sexp x = user[name];
  if (x != R_NilValue) {
    if (object_length(x) != 1) {
      cpp11::stop("Expected a scalar numeric for '%s'", name);
    }
    // TODO: when we're getting out an integer this is a bit too relaxed
    if (TYPEOF(x) == REALSXP) {
      ret = cpp11::as_cpp<T>(x);
    } else if (TYPEOF(x) == INTSXP) {
      ret = cpp11::as_cpp<T>(x);
    } else {
      cpp11::stop("Expected a numeric value for %s", name);
    }
  }

  if (is_na(ret)) {
    cpp11::stop("Expected a value for '%s'", name);
  }
  user_check_value<T>(ret, name, min, max);
  return ret;
}

// This is not actually really enough to work generally as there's an
// issue with what to do with checking previous, min and max against
// NA_REAL -- which is not going to be the correct value for float
// rather than double.  Further, this is not extendable to the vector
// cases because we hit issues around partial template specification.
//
// We can make the latter go away by replacing std::array<T, N> with
// std::vector<T> - the cost is not great.  But the NA issues remain
// and will require further thought. However, this template
// specialisation and the tests that use it ensure that the core code
// generation is at least compatible with floats.
//
// See #6
template <>
inline float user_get_scalar<float>(cpp11::list user, const char *name,
                                    const float previous, float min, float max) {
  double value = user_get_scalar<double>(user, name, previous, min, max);
  return static_cast<float>(value);
}

template <typename T, size_t N>
std::vector<T> user_get_array_fixed(cpp11::list user, const char *name,
                                    const std::vector<T> previous,
                                    const std::array<int, N>& dim,
                                    T min, T max) {
  cpp11::sexp x = user[name];
  if (x == R_NilValue) {
    if (previous.size() == 0) {
      cpp11::stop("Expected a value for '%s'", name);
    }
    return previous;
  }

  user_check_array_rank<N>(x, name);
  user_check_array_dim<N>(x, name, dim);

  std::vector<T> ret = cpp11::as_cpp<std::vector<T>>(x);
  user_check_array_value(ret, name, min, max);

  return ret;
}

template <typename T, size_t N>
std::vector<T> user_get_array_variable(cpp11::list user, const char *name,
                                       std::vector<T> previous,
                                       std::array<int, N>& dim,
                                       T min, T max) {
  cpp11::sexp x = user[name];
  if (x == R_NilValue) {
    if (previous.size() == 0) {
      cpp11::stop("Expected a value for '%s'", name);
    }
    return previous;
  }

  user_check_array_rank<N>(x, name);
  user_set_array_dim<N>(x, name, dim);

  std::vector<T> ret = cpp11::as_cpp<std::vector<T>>(x);
  user_check_array_value(ret, name, min, max);

  return ret;
}

// This is sum with inclusive "from", exclusive "to", following the
// same function in odin
template <typename T>
T odin_sum1(const T * x, size_t from, size_t to) {
  T tot = 0.0;
  for (size_t i = from; i < to; ++i) {
    tot += x[i];
  }
  return tot;
}
template<>
sirs::init_t dust_data<sirs>(cpp11::list user) {
  typedef typename sirs::real_t real_t;
  sirs::init_t internal;
  internal.initial_R = 0;
  internal.I_ini = NA_REAL;
  internal.alpha = 0.10000000000000001;
  internal.beta = 0.20000000000000001;
  internal.gamma = 0.10000000000000001;
  internal.S_ini = 1000;
  internal.alpha = user_get_scalar<real_t>(user, "alpha", internal.alpha, NA_REAL, NA_REAL);
  internal.beta = user_get_scalar<real_t>(user, "beta", internal.beta, NA_REAL, NA_REAL);
  internal.gamma = user_get_scalar<real_t>(user, "gamma", internal.gamma, NA_REAL, NA_REAL);
  internal.I_ini = user_get_scalar<real_t>(user, "I_ini", internal.I_ini, NA_REAL, NA_REAL);
  internal.S_ini = user_get_scalar<real_t>(user, "S_ini", internal.S_ini, NA_REAL, NA_REAL);
  internal.initial_I = internal.I_ini;
  internal.initial_S = internal.S_ini;
  internal.p_IR = 1 - std::exp(- internal.gamma);
  internal.p_RS = 1 - std::exp(- internal.alpha);
  return internal;
}
template <>
cpp11::sexp dust_info<sirs>(const sirs::init_t& internal) {
  cpp11::writable::list ret(3);
  ret[0] = cpp11::writable::integers({1});
  ret[1] = cpp11::writable::integers({1});
  ret[2] = cpp11::writable::integers({1});
  cpp11::writable::strings nms({"S", "I", "R"});
  ret.names() = nms;
  return ret;
}