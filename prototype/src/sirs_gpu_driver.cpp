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

[[cpp11::register]]
SEXP dust_sirs_alloc(cpp11::list r_data, size_t step, size_t n_particles,
                size_t n_threads, size_t seed) {
  validate_size(step, "step");
  validate_size(n_particles, "n_particles");
  validate_size(n_threads, "n_threads");
  validate_size(seed, "seed");
  typename T::init_t data = dust_data<T>(r_data);

  void *d = SOMETHING(data, step, n_particles, n_threads, seed);
  // TODO: this should get a deleter added!
  cpp11::external_pointer<void*> ptr(d, false, true);
  cpp11::sexp info = dust_info<T>(data);

  return cpp11::writable::list({ptr, info});
}
