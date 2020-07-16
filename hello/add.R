dyn.load("add.so")

add_gpu <- function(a, b) {
  n <- length(a)
  stopifnot(length(b) == n)
  .C("gvectorAdd", as.double(a), as.double(b), double(n), as.integer(n),
     PACKAGE = "add")[[3L]]
}

add_c <- function(a, b) {
  n <- length(a)
  stopifnot(length(b) == n)
  .C("add_c", as.double(a), as.double(b), as.integer(n), double(n),
     PACKAGE = "add")[[4L]]
}

add_call <- function(a, b) {
  stopifnot(length(b) == length(a))
  .Call("add_call", as.double(a), as.double(b))
}

a <- runif(1000)
b <- runif(1000)

value_cpu <- a + b
value_gpu <- add_gpu(a, b)
identical(value_cpu, value_gpu)
identical(value_gpu, add_c(a, b))
identical(value_gpu, add_call(a, b))
