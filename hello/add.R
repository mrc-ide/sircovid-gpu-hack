dyn.load("add.so")

add_gpu <- function(a, b) {
  n <- length(x)
  stopifnot(length(b) == n)
  .C("add", as.double(a), as.double(b), double(n), PACKAGE = "add")[[3L]]
}

a <- runif(1000)
b <- runif(1000)

value_cpu <- a + b
value_gpu <- add_gpu(a, b)
identical(value_cpu, value_gpu)
