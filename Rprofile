options(repos = "https://cloud.r-project.org/",
        browserNLdisabled = TRUE,
        menu.graphics = FALSE,
        warnPartialMatchAttr = TRUE,
        warnPartialMatchDollar = TRUE,
        warnPartialMatchArgs = TRUE)

attach(local({
  .test_package <- function(pkg = ".", filter = NULL, ...) {
    root <- rprojroot::find_package_root_file(path = pkg)
    package <- unname(desc::desc_get("Package", root))
    test_path <- file.path(root, "tests", "testthat")

    ns_env <- pkgload::load_all(root, quiet = TRUE)$env
    env <- new.env(parent = ns_env)

    testthat_args <- list(test_path, filter = filter, env = env,
                          load_helpers = FALSE, stop_on_failure = FALSE,
                          ... = ...)

    envvar <- c(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep),
                CYGWIN = "nodosfilewarning", R_TESTS = "", R_BROWSER = "false",
                R_PDFVIEWER = "false", NOT_CRAN = "true",
                TESTTHAT_PKG = package)

    withr::with_options(
      c(useFancyQuotes = FALSE),
      withr::with_envvar(
        envvar,
        do.call(testthat::test_dir, testthat_args)))
  }

  environment()
}),
name = ".Rprofile")
