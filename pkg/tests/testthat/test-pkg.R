context("pkg")

test_that("add_r agrees with R's add", {
  a <- runif(10)
  b <- runif(10)
  expect_equal(add_r(a, b), a + b, tolerance = 1e-13)
})


test_that("fft_r agrees with R's fft", {
  z <- complex(real = rnorm(10), imaginary = rnorm(10))
  expect_equal(fft_r(z), fft(z), tolerance = 1e-13)
  expect_equal(fft_r(z, inverse = TRUE), fft(z, inverse = TRUE),
               tolerance = 1e-13)
})

test_that("add_c agrees with add_r", {
  a <- runif(10)
  b <- runif(10)
  expect_identical(add_c(a, b), add_r(a, b))
})

test_that("add_call agrees with add_r", {
  a <- runif(10)
  b <- runif(10)
  expect_identical(add_call(a, b), add_r(a, b))
})
