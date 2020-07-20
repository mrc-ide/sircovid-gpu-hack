#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include "sirs_gpu.hpp"

[[cpp11::register]]
cpp11::writable::doubles prototype(double alpha, double beta, double gamma,
                                   int I0, int n_particles, int n_steps,
                                   int n_record, int seed) {
  auto value = sircovid_main(alpha, beta, gamma, I0, n_particles, n_steps,
                             n_record, seed);

  cpp11::writable::doubles ret(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    ret[i] = value[i];
  }

  const int n_state = 3;
  ret.attr("dim") =
    cpp11::writable::integers({n_state, n_particles, n_steps + 1});
  return ret;
}
