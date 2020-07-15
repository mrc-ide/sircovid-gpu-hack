## MRCGIDA GPU Hackathon Notes

The hackathon raplab docs explain getting started and are posted in the help channel. This doc picks up assuming all the Okta bits are set up and tested.

Log on to the cluster with:

```
sft ssh raplab-hackathon
```

This does a little dance with your browser (and possibly your phone!) and launches a website that you can ignore once the terminal has connected.

The head nodes and compute nodes have different architectures so do not work on the head node. Start an interactive task with:

```
srun --ntasks=5 --nodes=1 --cpus-per-task=2 --partition=batch --time=4:00:00 --gres=gpu:1 --pty /bin/bash
```

or if you do not need the GPU:

```
srun --ntasks=5 --nodes=1 --cpus-per-task=2 --partition=batch --time=4:00:00 --pty /bin/bash
```

Which starts an interactive job and sets you up with a terminal there. It will look much the same but the node name will have changed from `dgx0181` to `cpu0123` -- numbers will probably vary!

We need to load the R modules - the organisers have set us up with 4.0.2 which is the same as travis is using

```
module load Core/gcc/8.4.0 gcc/8.4.0/r/4.0.2
```

It's not clear yet if we share a workspace or not, but installing packages takes a while.

When installing things the first time you will be told:

```
Warning in install.packages("remotes") :
  'lib = "/mnt/shared/sw-hackathons/spack-0.15.1/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/r-4.0.2-nael6nmjxfamq2dy2cpv3c36ypf6fefs/rlib/R/library"' is not writable
Would you like to use a personal library instead? (yes/No/cancel)
```

Answer `yes` and it will prompt you:

```
Would you like to create a personal library
‘~/R/x86_64-pc-linux-gnu-library/4.0’
to install packages into? (yes/No/cancel)
```

answer `yes` again and things will be set up well now and for future sessions (and you will not be prompted again).

To install everything, including dependencies:

```r
install.packages("remotes")
remotes::install_github("mrc-ide/dust", dependencies = TRUE)
```

(this will take a while, beware)

The `devtools` package can't be easily installed as it requires curl headers. We could probably get this installed but it will not be needed for most of what we are doing. Instead, try this wrapper:

```r
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
```

Automate all of this by running:

```
curl -LO
```

You can clone down a recent copy of `dust` with:

```
git clone https://github.com/mrc-ide/dust
```

Confirm everything works by compiling `dust` and running its tests:

```r
.test_package("dust")
```
