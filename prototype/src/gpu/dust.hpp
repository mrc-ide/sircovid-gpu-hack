#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

// Not sure if we want the RNG object in CUDA
// #include "rng.hpp"
#include "gpu/xoshiro.hpp"
#include "gpu/distr/binomial.hpp"

#include <algorithm>
#include <utility>
#ifdef _OPENMP
#if _OPENMP >= 201511
#define OPENMP_HAS_MONOTONIC 1
#endif
#include <omp.h>
#endif

#include <thrust/device_vector.h>
#include <thrust/swap.h>

// Error checking of dynamic memory allocation on device
// https://stackoverflow.com/a/14038590
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__host__ __device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

template <typename T>
__global__
void run_particles(T* model,
                  real_t** particle_y,
                  real_t** particle_y_swap,
                  uint64_t* rng_state,
                  size_t y_len,
                  size_t n_particles,
                  size_t step,
                  size_t step_end) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int p_idx = index; p_idx < n_particles; p_idx += stride) {
    int curr_step = step;
    while (curr_step < step_end) {
      model->update(curr_step,
                    particle_y[p_idx],
                    rng_state + p_idx * XOSHIRO_WIDTH,
                    particle_y_swap[p_idx]);
      curr_step++;
      real_t* tmp = particle_y[p_idx];
      particle_y[p_idx] = particle_y_swap[p_idx];
      particle_y_swap[p_idx] = tmp;
    }
  }
}

template <typename T>
class Particle {
public:
  typedef typename T::init_t init_t;
  typedef typename T::int_t int_t;
  typedef typename T::real_t real_t;
  // typedef typename dust::RNG<real_t, int_t> rng_t;

  Particle(init_t data, size_t step) :
    _model(data),
    _step(step),
    _y(_model.initial(_step)),
    _y_swap(_model.size()) {
      cdpErrchk(cudaMalloc((void** )&_y_device, _y.size() * sizeof(real_t)));
      cdpErrchk(cudaMemcpy(_y_device, _y.data(), _y.size() * sizeof(real_t),
              cudaMemcpyDefault));
      cdpErrchk(cudaMalloc((void** )&_y_swap_device, _y_swap.size() * sizeof(real_t)));
      cdpErrchk(cudaMemcpy(_y_swap_device, _y_swap.data(), _y_swap.size() * sizeof(real_t),
              cudaMemcpyDefault));
      cudaDeviceSynchronize();
  }

  ~Particle() {
    cdpErrchk(cudaFree(_y_device));
    cdpErrchk(cudaFree(_y_swap_device));
  }

  Particle(Particle&& other) noexcept :
    _model(std::move(other._model)),
    _step(std::move(other._step)),
    _y(std::move(other._y)),
    _y_swap(std::move(other._y_swap)),
    _y_device(std::exchange(other._y_device, nullptr)),
    _y_swap_device(std::exchange(other._y_swap_device, nullptr))
  {}

  Particle& operator=(Particle&& other) {
    if (this != &other) {
      cdpErrchk(cudaFree(_y_device));
      cdpErrchk(cudaFree(_y_swap_device));

      std::swap(_model, other._model);
      std::swap(_step, other._step);
      std::swap(_y, other._y);
      std::swap(_y_swap, other._y_swap);
      _y_device = std::exchange(other._y_device, nullptr);
      _y_swap_device = std::exchange(other._y_swap_device, nullptr);
    }
    return *this;
  }

  real_t * y_addr() { return _y_device; };
  real_t * y_swap_addr() { return _y_swap_device; };

  void state(const std::vector<size_t>& index_y,
             typename std::vector<real_t>::iterator end_state) {
    // TODO: efficiency of copying whole state each time, when only some of it
    // is used? Random access would be better, if possible
    y_to_host();
    for (size_t i = 0; i < index_y.size(); ++i) {
      *(end_state + i) = _y[index_y[i]];
    }
  }

  void state_full(typename std::vector<real_t>::iterator end_state) {
    y_to_host();
    for (size_t i = 0; i < _y.size(); ++i) {
      *(end_state + i) = _y[i];
    }
  }

  size_t size() const {
    return _y.size();
  }

  void set_step(size_t step) {
    _step = step;
  }

  size_t step() const {
    return _step;
  }

  void swap() {
    thrust::device_ptr<real_t> y_ptr(_y_device);
    thrust::device_ptr<real_t> y_swap_ptr(_y_swap_device);
    thrust::swap(y_ptr, y_swap_ptr);
    // Necessary to swap on host too?:
    /*
    thrust::copy(_y_device.begin(), _y_device.end(), _y.begin());
    thrust::copy(_y_swap_device.begin(), _y_swap_device.end(), _y_swap.begin());
    thrust::swap(_y, _y_swap);
    cudaDeviceSynchronize();
    */
  }

  void set_state(const Particle<T>& other) {
    _y_swap = other._y;
    y_swap_to_device();
  }

  void set_state(typename std::vector<real_t>::const_iterator state) {
    for (size_t i = 0; i < _y.size(); ++i, ++state) {
      _y[i] = *state;
    }
    y_to_device();
  }

private:
  // Delete copy
  Particle ( const Particle & ) = delete;

  T _model;
  size_t _step;

  std::vector<real_t> _y;
  std::vector<real_t> _y_swap;
  real_t * _y_device;
  real_t * _y_swap_device;

  void y_to_host() {
    cdpErrchk(cudaMemcpy(_y.data(), _y_device, _y.size() * sizeof(real_t),
              cudaMemcpyDefault));
    cudaDeviceSynchronize();
  }
  void y_swap_to_device() {
    cdpErrchk(cudaMemcpy(_y_swap_device, _y_swap.data(), _y_swap.size() * sizeof(real_t),
              cudaMemcpyDefault));
    cudaDeviceSynchronize();
  }
  void y_to_device() {
    cdpErrchk(cudaMemcpy(_y_device, _y.data(), _y.size() * sizeof(real_t),
              cudaMemcpyDefault));
    cudaDeviceSynchronize();
  }
};

template <typename T>
class Dust {
public:
  typedef typename T::init_t init_t;
  typedef typename T::int_t int_t;
  typedef typename T::real_t real_t;
  // typedef typename dust::RNG<real_t, int_t> rng_t;

  Dust(const init_t data, const size_t step, const size_t n_particles,
       const size_t n_threads, const size_t seed) :
    _n_threads(n_threads) {
    initialise(data, step, n_particles);

    // Set up rng streams for each particle
    cdpErrchk(cudaMallocManaged((void** )&_rng_state, n_particles * XOSHIRO_WIDTH * sizeof(uint64_t)));
    dust::Xoshiro rng(seed);
    for (int i = 0; i < n_particles; i++) {
      uint64_t* current_state = rng.get_rng_state();
      for (int state_idx = 0; state_idx < XOSHIRO_WIDTH; state_idx++) {
        _rng_state[i * XOSHIRO_WIDTH + state_idx] = current_state[state_idx];
      }
      rng.jump();
    }
    cudaDeviceSynchronize();
  }

  ~Dust() {
    cdpErrchk(cudaFree(_model));
    cdpErrchk(cudaFree(_rng_state));
    cdpErrchk(cudaFree(_particle_y_addrs));
    cdpErrchk(cudaFree(_particle_y_swap_addrs));
  }

  void reset(const init_t data, const size_t step) {
    const size_t n_particles = _particles.size();
    initialise(data, step, n_particles);
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    _index = index;
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_t>& state, bool is_matrix) {
    const size_t n_particles = _particles.size();
    const size_t n_state = n_state_full();
    auto it = state.begin();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_state(it);
      if (is_matrix) {
        it += n_state;
      }
    }
  }

  void set_step(const size_t step) {
    const size_t n_particles = _particles.size();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_step(step);
    }
  }

  void set_step(const std::vector<size_t>& step) {
    const size_t n_particles = _particles.size();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_step(step[i]);
    }
    const auto r = std::minmax_element(step.begin(), step.end());
    if (*r.second > *r.first) {
      run(*r.second);
    }
  }

  void run(const size_t step_end) {
    const size_t blockSize = 32; // Check later
    const size_t blockCount = (_particles.size() + blockSize - 1) / blockSize;
    // const size_t blockSize = 1;
    // const size_t blockCount = 1;
    run_particles<<<blockCount, blockSize>>>(_model,
                                            _particle_y_addrs,
                                            _particle_y_swap_addrs,
                                            _rng_state,
                                            _model->size(),
                                            _particles.size(),
                                            this->step(),
                                            step_end);
    // write step end back to particles
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].set_step(step_end);
    }
    cudaDeviceSynchronize();
  }

  void state(std::vector<real_t>& end_state) {
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(_index, end_state.begin() + i * _index.size());
    }
  }

  void state(std::vector<size_t> index_y,
             std::vector<real_t>& end_state) {
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(index_y, end_state.begin() + i * index_y.size());
    }
  }

  void state_full(std::vector<real_t>& end_state) {
    const size_t n = n_state_full();
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state_full(end_state.begin() + i * n);
    }
  }

  // There are two obvious ways of reordering; we can construct a
  // completely new set of particles, like
  //
  //   std::vector<Particle<T>> next;
  //   for (auto const& i: index) {
  //     next.push_back(_particles[i]);
  //   }
  //   _particles = next;
  //
  // but this seems like a lot of churn.  The other way is to treat it
  // like a slightly weird state update where we swap around the
  // contents of the particle state (uses the update() and swap()
  // methods on particles).
  void reorder(const std::vector<size_t>& index) {
    for (size_t i = 0; i < _particles.size(); ++i) {
      size_t j = index[i];
      _particles[i].set_state(_particles[j]);
    }
    for (auto& p : _particles) {
      p.swap();
    }
  }

  size_t n_particles() const {
    return _particles.size();
  }

  size_t n_state() const {
    return _index.size();
  }

  size_t n_state_full() const {
    return _particles.front().size();
  }

  size_t step() const {
    return _particles.front().step();
  }

private:
  // delete move and copy to avoid accidentally using them
  Dust ( const Dust & ) = delete;
  Dust ( Dust && ) = delete;

  std::vector<size_t> _index;
  const size_t _n_threads;
  //dust::pRNG<real_t, int_t> _rng;
  std::vector<Particle<T>> _particles;

  T* _model;
  real_t** _particle_y_addrs;
  real_t** _particle_y_swap_addrs;
  uint64_t* _rng_state;

  void initialise(const init_t data, const size_t step,
                  const size_t n_particles) {
    _particles.clear();
    _particles.reserve(n_particles);

    cudaFree(_particle_y_addrs);
    cudaFree(_particle_y_swap_addrs);
    cudaFree(_model);

    std::vector<real_t*> y_ptrs;
    std::vector<real_t*> y_swap_ptrs;
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
      y_ptrs.push_back(_particles[i].y_addr());
      y_swap_ptrs.push_back(_particles[i].y_swap_addr());
    }
    cdpErrchk(cudaMalloc((void** )&_particle_y_addrs, y_ptrs.size() * sizeof(real_t*)));
    cdpErrchk(cudaMemcpy(_particle_y_addrs, y_ptrs.data(), y_ptrs.size() * sizeof(real_t*),
	              cudaMemcpyHostToDevice));
    cdpErrchk(cudaMalloc((void** )&_particle_y_swap_addrs, y_swap_ptrs.size() * sizeof(real_t*)));
    cdpErrchk(cudaMemcpy(_particle_y_swap_addrs, y_swap_ptrs.data(), y_swap_ptrs.size() * sizeof(real_t*),
	              cudaMemcpyHostToDevice));

    // Copy the model
    cdpErrchk(cudaMallocManaged((void** )&_model, sizeof(T)));
    *_model = T(data);

    const size_t n = n_state_full();
    _index.clear();
    _index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      _index.push_back(i);
    }
  }
};

#endif
