#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include "xoshiro.hpp"
#include "distr/binomial.hpp"
#include "distr/normal.hpp"
#include "distr/poisson.hpp"
#include "distr/uniform.hpp"

namespace dust {

// Read state from global memory
__device__
RNGState loadRNG(uint64_t * rng_state, int p_idx, size_t n_particles) {
  RNGState state;
  state.s0 = rng_state[p_idx];
  state.s1 = rng_state[p_idx + n_particles];
  state.s2 = rng_state[p_idx + n_particles * 2];
  state.s3 = rng_state[p_idx + n_particles * 3];
  return state;
}

// Write state into global memory
__device__
void putRNG(RNGState& rng, uint64_t* rng_state, int p_idx, size_t n_particles) {
  rng_state[p_idx] = rng.s0;
  rng_state[p_idx + n_particles] = rng.s1;
  rng_state[p_idx + n_particles * 2] = rng.s2;
  rng_state[p_idx + n_particles * 3] = rng.s3;
}

template <typename real_t, typename int_t>
class pRNG { // # nocov
public:
  pRNG(const size_t n, const uint64_t seed) {
    dust::Xoshiro rng(seed);
    std::vector<dust::distr::rnorm<real_t>> rnorm_buffers;
    for (int i = 0; i < n; i++) {
      rnorm_buffers.push_back(dust::distr::rnorm<real_t>());
      _rngs.push_back(rng);
      rng.jump();
    }

    cdpErrchk(cudaMalloc((void** )&_rnorm_buffers,
                         n * sizeof(dust::distr::rnorm<real_t>)));
    cdpErrchk(cudaMemcpy(_rnorm_buffers, rnorm_buffers.data(),
                         n * sizeof(dust::distr::rnorm<real_t>),
                         cudaMemcpyDefault));

    cdpErrchk(cudaMalloc((void** )&_rng_state,
                          n * XOSHIRO_WIDTH * sizeof(uint64_t)));
    put_state();
  }

  ~pRNG() {
    cdpErrchk(cudaFree(_rng_state));
    cdpErrchk(cudaFree(_rnorm_buffers));
  }

  uint64_t* state_ptr() { return _d_rng_state; }
  dust::distr::rnorm<real_t>* rnorm_ptr() { return _rnorm_buffers; }

  size_t size() const {
    return _rngs.size();
  }

  void jump() {
    get_state();
    for (size_t i = 0; i < _rngs.size(); ++i) {
      _rngs[i].jump();
    }
    put_state();
  }

  void long_jump() {
    get_state();
    for (size_t i = 0; i < _rngs.size(); ++i) {
      _rngs[i].long_jump();
    }
    put_state();
  }

private:
  // delete move and copy to avoid accidentally using them
  pRNG ( const pRNG & ) = delete;
  pRNG ( pRNG && ) = delete;

  void put_state() {
    std::vector<uint64_t> interleaved_state(n * XOSHIRO_WIDTH);
    for (int i = 0; i < n; i++) {
      uint64_t* current_state = _rngs[i].get_rng_state();
      for (int state_idx = 0; state_idx < XOSHIRO_WIDTH; state_idx++) {
        interleaved_state[i + n * state_idx] = current_state[state_idx];
      }
    }
    cdpErrchk(cudaMemcpy(_d_rng_state, interleaved_state.data(),
                         interleaved_state.size() * sizeof(uint64_t),
                         cudaMemcpyDefault));
    cudaDeviceSynchronize();
  }

  void get_state() {
    std::vector<uint64_t> interleaved_state(n * XOSHIRO_WIDTH);
    cdpErrchk(cudaMemcpy(interleaved_state.data(), _d_rng_state,
                         interleaved_state.size() * sizeof(uint64_t),
                         cudaMemcpyDefault));
    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++) {
      std::vector<uint64_t> state(XOSHIRO_WIDTH);
      for (int state_idx = 0; state_idx < XOSHIRO_WIDTH; state_idx++) {
        state[i] = interleaved_state[i + n * state_idx];
      }
      _rngs[i].set_state(state)
    }
  }

  // Host memory
  std::vector<dust::Xoshiro<real_t>> _rngs;

  // Device memory
  uint64_t* _d_rng_state;
  dust::distr::rnorm<real_t>* _rnorm_buffers;
};

#endif
