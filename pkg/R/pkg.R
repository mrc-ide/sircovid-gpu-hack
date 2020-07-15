## Completely trivial R functions for scaffolding
add_r <- function(a, b) {
  a + b
}

fft_r <- function(z, inverse = FALSE) {
  stats::fft(z, inverse = inverse)
}
