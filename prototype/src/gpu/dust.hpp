#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include "rng.hpp"

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
// #include <cub/device/device_select.cuh>

template <typename T>
__global__
void run_particles(T** models,
                  real_t** particle_y,
                  real_t** particle_y_swap,
                  dust::RNGptr rng_state,
                  size_t y_len,
                  size_t n_particles,
                  size_t step,
                  size_t step_end) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int p_idx = index; p_idx < n_particles; p_idx += stride) {
    dust::RNGState rng = dust::loadRNG(rng_state, p_idx);
    int curr_step = step;
    while (curr_step < step_end) {
      // Run the model forward a step
      models[p_idx]->update(curr_step,
                            particle_y[p_idx],
                            particle_y_swap[p_idx],
                            rng);
      __syncwarp();
      curr_step++;

      // Update state
      real_t* tmp = particle_y[p_idx];
      particle_y[p_idx] = particle_y_swap[p_idx];
      particle_y_swap[p_idx] = tmp;
    }
    dust::putRNG(rng, rng_state, p_idx);
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
    _step(step) {
      // Copy the model
      CUDA_CALL(cudaMallocManaged((void** )&_model, sizeof(T)));
      *_model = T(data);
      cudaDeviceSynchronize();

      _y = std::vector<real_t>(_model->initial(_step));
      _y_swap = std::vector<real_t>(_model->size());

      CUDA_CALL(cudaMalloc((void** )&_y_device, _y.size() * sizeof(real_t)));
      CUDA_CALL(cudaMemcpy(_y_device, _y.data(), _y.size() * sizeof(real_t),
              cudaMemcpyDefault));
      CUDA_CALL(cudaMalloc((void** )&_y_swap_device, _y_swap.size() * sizeof(real_t)));
      CUDA_CALL(cudaMemcpy(_y_swap_device, _y_swap.data(), _y_swap.size() * sizeof(real_t),
              cudaMemcpyDefault));
  }

  ~Particle() {
    CUDA_CALL(cudaFree(_y_device));
    CUDA_CALL(cudaFree(_y_swap_device));
    CUDA_CALL(cudaFree(_model));
  }

  Particle(Particle&& other) noexcept :
    _step(std::move(other._step)),
    _y(std::move(other._y)),
    _y_swap(std::move(other._y_swap)),
    _y_device(nullptr),
    _y_swap_device(nullptr),
    _model(nullptr)
  {
    _y_device = other._y_device;
    other._y_device = nullptr;
    _y_swap_device = other._y_swap_device;
    other._y_swap_device = nullptr;
    _model = other._model;
    other._model = nullptr;
  }

  Particle& operator=(Particle&& other) {
    if (this != &other) {
      CUDA_CALL(cudaFree(_y_device));
      CUDA_CALL(cudaFree(_y_swap_device));

      std::swap(_model, other._model);
      std::swap(_step, other._step);
      std::swap(_y, other._y);
      std::swap(_y_swap, other._y_swap);
      _y_device = other._y_device;
      other._y_device = nullptr;
      _y_swap_device = other._y_swap_device;
      other._y_swap_device = nullptr;
      _model = other._model;
      other._model = nullptr;
    }
    return *this;
  }

  real_t * y_addr() { return _y_device; };
  real_t * y_swap_addr() { return _y_swap_device; };
  T * model_addr() { return _model; }

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
    // Swaps on the device
    thrust::device_ptr<real_t> y_ptr(_y_device);
    thrust::device_ptr<real_t> y_swap_ptr(_y_swap_device);
    thrust::swap(y_ptr, y_swap_ptr);
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

  T* _model;
  size_t _step;

  std::vector<real_t> _y;
  std::vector<real_t> _y_swap;
  real_t * _y_device;
  real_t * _y_swap_device;

  void y_to_host() {
    CUDA_CALL(cudaMemcpy(_y.data(), _y_device, _y.size() * sizeof(real_t),
              cudaMemcpyDefault));
    cudaDeviceSynchronize();
  }
  void y_swap_to_device() {
    CUDA_CALL(cudaMemcpy(_y_swap_device, _y_swap.data(), _y_swap.size() * sizeof(real_t),
              cudaMemcpyDefault));
    cudaDeviceSynchronize();
  }
  void y_to_device() {
    CUDA_CALL(cudaMemcpy(_y_device, _y.data(), _y.size() * sizeof(real_t),
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
    _n_threads(n_threads),
    _rng(n_particles, seed),
    _model_addrs(nullptr),
    _particle_y_addrs(nullptr),
    _particle_y_swap_addrs(nullptr) {
    initialise(data, step, n_particles);
    cudaDeviceSynchronize();
  }

  // NB - if you call cudaDeviceReset() this destructor will segfault
  ~Dust() {
    CUDA_CALL(cudaFree(_model_addrs));
    CUDA_CALL(cudaFree(_particle_y_addrs));
    CUDA_CALL(cudaFree(_particle_y_swap_addrs));
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
    run_particles<<<blockCount, blockSize>>>(_model_addrs,
                                            _particle_y_addrs,
                                            _particle_y_swap_addrs,
                                            _rng.state_ptr(),
                                            _particles.front().size(), //FIXME: are sizes ever different?
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
  dust::pRNG<real_t, int_t> _rng;
  std::vector<Particle<T>> _particles;

  T** _model_addrs;
  real_t** _particle_y_addrs;
  real_t** _particle_y_swap_addrs;

  void initialise(const init_t data, const size_t step,
                  const size_t n_particles) {
    _particles.clear();
    _particles.reserve(n_particles);

    CUDA_CALL(cudaFree(_particle_y_addrs));
    CUDA_CALL(cudaFree(_particle_y_swap_addrs));
    CUDA_CALL(cudaFree(_model_addrs));

    std::vector<real_t*> y_ptrs;
    std::vector<real_t*> y_swap_ptrs;
    std::vector<T*> model_ptrs;
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
      y_ptrs.push_back(_particles[i].y_addr());
      y_swap_ptrs.push_back(_particles[i].y_swap_addr());
      model_ptrs.push_back(_particles[i].model_addr());
    }
    CUDA_CALL(cudaMalloc((void** )&_particle_y_addrs, y_ptrs.size() * sizeof(real_t*)));
    CUDA_CALL(cudaMemcpy(_particle_y_addrs, y_ptrs.data(), y_ptrs.size() * sizeof(real_t*),
	              cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void** )&_particle_y_swap_addrs, y_swap_ptrs.size() * sizeof(real_t*)));
    CUDA_CALL(cudaMemcpy(_particle_y_swap_addrs, y_swap_ptrs.data(), y_swap_ptrs.size() * sizeof(real_t*),
	              cudaMemcpyHostToDevice));

    // Copy the model
    CUDA_CALL(cudaMalloc((void** )&_model_addrs, model_ptrs.size() * sizeof(T*)));
    CUDA_CALL(cudaMemcpy(_model_addrs, model_ptrs.data(), model_ptrs.size() * sizeof(T*),
	              cudaMemcpyHostToDevice));

    const size_t n = n_state_full();
    _index.clear();
    _index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      _index.push_back(i);
    }
  }
};

#endif
