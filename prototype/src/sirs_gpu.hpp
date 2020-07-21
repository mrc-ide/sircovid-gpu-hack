#include <vector>
std::vector<float> sircovid_main(float alpha, float beta, float gamma, int I0,
                                 int n_particles, int n_steps, int n_record,
                                 int seed);

// These only exist so that cpp11 finds them as it can't look within
// .cu files
#ifndef __NVCC__
[[cpp11::register]]
#endif
SEXP dust_sireinfect_alloc(cpp11::list r_data, size_t step, size_t n_particles,
                           size_t n_threads, size_t seed);

#ifndef __NVCC__
[[cpp11::register]]
#endif
SEXP dust_sireinfect_run(SEXP ptr, size_t step_end);

#ifndef __NVCC__
[[cpp11::register]]
#endif
SEXP dust_sireinfect_set_index(SEXP ptr, cpp11::sexp r_index);

#ifndef __NVCC__
[[cpp11::register]]
#endif
SEXP dust_sireinfect_set_state(SEXP ptr, SEXP r_state, SEXP r_step);

#ifndef __NVCC__
[[cpp11::register]]
#endif
SEXP dust_sireinfect_reset(SEXP ptr, cpp11::list r_data, size_t step);

#ifndef __NVCC__
[[cpp11::register]]
#endif
SEXP dust_sireinfect_state(SEXP ptr, SEXP r_index);

#ifndef __NVCC__
[[cpp11::register]]
#endif
size_t dust_sireinfect_step(SEXP ptr);

#ifndef __NVCC__
[[cpp11::register]]
#endif
void dust_sireinfect_reorder(SEXP ptr, cpp11::sexp r_index);
