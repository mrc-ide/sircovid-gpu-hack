// Generated by dust (version 0.0.7) - do not edit
#include "sirs_gpu.hpp"
#include "gpu/dust.hpp"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

// Generated by odin.dust (version 0.0.3) - do not edit
class sireinfect {
public:
  typedef int int_t;
  typedef float real_t;
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
  sireinfect(const init_t& data): internal(data) {
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
  __device__
  void update(size_t step, const real_t * state, uint64_t * rng_state, real_t * state_next) {
    // TODO: state and state_next will not be coalesced read/write
    // A 2D array, transposed, would be better
    // Check that rng_state ends up in L1 cache
    const real_t S = state[0];
    const real_t I = state[1];
    const real_t R = state[2];
    real_t N = S + I + R;
    real_t n_IR = dust::distr::rbinom<real_t, int_t>(rng_state, rintf(I), internal.p_IR);
    real_t n_RS = dust::distr::rbinom<real_t, int_t>(rng_state, rintf(R), internal.p_RS);
    //real_t p_SI = 1 - std::exp(- internal.beta * I / (real_t) N);
    // NB - this is specific to a float, need to think about this if real_t = double
    // exp should just be overloaded
    real_t p_SI = 1 - expf(- internal.beta * I / (real_t) N);
    real_t n_SI = dust::distr::rbinom<real_t, int_t>(rng_state, rintf(S), p_SI);
    // NB - Make sure that state is read only once, and state_next is written only once
    state_next[2] = R + n_IR - n_RS;
    state_next[1] = I + n_SI - n_IR;
    state_next[0] = S - n_SI + n_RS;
  }
private:
  init_t internal;
};

std::vector<float> sircovid_main(float alpha, float beta, float gamma, int I0,
                                 int n_particles, int n_steps, int n_record,
                                 int seed) {
  const int_t n_state = 3;
  int_t S_ini = 1000;
  int_t tau = n_record;

  // Run CUDA initialisation
  // Hard coded device 0
  cudaSetDevice(0);
  cudaDeviceReset();

  // Set up dust object
  sireinfect::init_t data;
  data.initial_R = 0;
  data.I_ini = I_ini;
  data.alpha = alpha;
  data.beta = beta;
  data.gamma = gamma;
  data.S_ini = S_ini;
  data.initial_I = data.I_ini;
  data.initial_S = data.S_ini;
  data.p_IR = 1 - std::exp(- data.gamma);
  data.p_RS = 1 - std::exp(- data.alpha);

  std::vector<size_t> index_y = {0};
  // Initial state, first step, index_y, n_particles, cpu_threads, seed
  Dust<sireinfect> dust_obj(data, 0, index_y, n_particles, 2, 1);

  // Run particles
  std::vector<real_t> state(dust_obj.n_particles() * dust_obj.n_state_full());

  std::vector<float> ret;
  for (size_t i = 0; i < state.size(); ++i) {
    ret.push_back(state[i]);
  }

  if (tau <= 0 || tau > n_steps) {
    dust_obj.run(n_steps);
    // cudaDeviceSynchronize();
    //printf("Run complete\n");

    dust_obj.state_full(state);
    //cudaDeviceSynchronize();
    //printf("State complete\n");
  } else {
    int_t step = tau;
    while (step < n_steps) {
      dust_obj.run(step);
      dust_obj.state_full(state);

      for (int i = 0; i < state.size(); ++i) {
        ret.push_back(state[i]);
      }

      step += tau;
    }
  }

  return ret;
}
