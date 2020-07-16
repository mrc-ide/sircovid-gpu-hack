## Completely trivial R functions for scaffolding
add_r <- function(a, b) {
  a + b
}

fft_r <- function(z, inverse = FALSE) {
  stats::fft(z, inverse = inverse)
}

add_c <- function(a, b) {
  n <- length(a)
  stopifnot(length(b) == n)
  .C(`_add_c`, as.double(a), as.double(b), as.integer(n), double(n))[[4]]
}

add_call <- function(a, b) {
  stopifnot(length(a) == length(b))
  .Call(`_add_call`, as.double(a), as.double(b))
}

add_gpu <- function(a, b) {
  n <- length(a)
  stopifnot(length(b) == n)
  .C(`_add_gpu`, as.double(a), as.double(b), as.integer(n), double(n))[[4]]
}

##' @useDynLib pkg, .registration = TRUE
NULL
