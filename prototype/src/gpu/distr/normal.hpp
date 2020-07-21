#ifndef DUST_DISTR_NORMAL_HPP
#define DUST_DISTR_NORMAL_HPP

#include <cmath>

namespace dust {
namespace distr {

template <typename real_t>
__device__
inline void BoxMuller(RNGState& rng_state, real_t* d0, real_t* d1) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  // We cannot mark "epsilon" as "static const" because NVCC would complain
  const double epsilon = 1.0e-7;
  double u1 = device_unif_rand(rng_state);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const double v1 = 2 * M_PI * device_unif_rand(rng_state);
  const double u2 = sqrt(-2.0 * log(u1));
  sincos(v1, d0, d1);
  *d0 *= u2;
  *d1 *= u2;
}

template <typename real_t>
class rnorm {
 public:
  __device__
  rnorm() : _buffered(false) {}

  __device__
  inline real_t operator()(RNGState& rng_state, real_t mean, real_t sd) {
    real_t z0;
    if (buffered) {
      buffered = false;
      z0 = result[1];
    } else {
      BoxMuller<real_t>(rng_state, &result[0], &result[1]);
      buffered = true;
      z0 = result[0];
    }
    __syncwarp();
    return(z0 * sigma + mu);
  }

  private:
    bool _buffered;
    real_t result[2];
};

}
}

#endif
