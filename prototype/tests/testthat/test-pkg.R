context("prototype")

test_that("reference example", {
  mod <- sirs$new(list(I_ini = 10), 0, 10)
  y <- mod$run(10)
  expect_equal(
    y[1, ],
    c(999, 958, 981, 983, 978, 976, 974, 967, 965, 976))
  expect_equal(
    y[2, ],
    c(6, 37, 20, 18, 21, 30, 28, 29, 26, 25))
  expect_equal(
    y[3, ],
    c(5, 15, 9, 9, 11, 4, 8, 14, 19, 9))
})
