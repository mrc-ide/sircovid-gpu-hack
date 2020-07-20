#include <vector>
std::vector<float> sircovid_main(float alpha, float beta, float gamma, int I0,
                                 int n_particles, int n_steps, int n_record,
                                 int seed);

template <typename T>
void* dust_gpu_alloc(typename T::init_t data, int step, int n_particles,
                     int n_threads, int seed) {
  // TODO: the gpu/dust.hpp code is out of date and so this needs
  // fixing!
  std::vector<size_t> index_y = {0};
  new dust_obj(data, 0, index_y, n_particles, n_threads, seed);

  Dust<T> *obj = new Dust<T>(data, step, n_particles, n_threads, seed);

  return (void*)obj;
}
