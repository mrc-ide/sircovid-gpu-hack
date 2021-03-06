## Manually compiling a shared library

This is the simplest task - can we get a shared library into R and call it. The approach is taken from [this blog post](https://developer.nvidia.com/blog/accelerate-r-applications-cuda/)

It is *not* general and we'll need to do some tricks to get it to work within the usual R CMD SHLIB approach (it's not clear that the linking approach here will always work, in particular).

Compile the programs with

```
./compile.sh
```

Then run the two example programs with:

```
Rscript cufft.R
Rscript add.R
```

both of which should print

```
[1] TRUE
```

if the GPU and CPU versions agree.
