dyn.load("cufft.so")

fft_gpu <- function(x, inverse = FALSE) {
  n <- length(x)
  rst <- .C("cufft",
            as.integer(n),
            as.integer(inverse),
            as.double(Re(z)),
            as.double(Im(z)),
            re = double(length = n),
            im = double(length = n),
            PACKAGE = "cufft")
  complex(real = rst[["re"]], imaginary = rst[["im"]])
}

num <- 4
z <- complex(real = stats::rnorm(num), imaginary = stats::rnorm(num))

value_cpu <- fft(z)
value_gpu <- fft_gpu(z)
all.equal(value_cpu, value_gpu, tolerance = 1e-13)
