# Fake Algo

Makes all the calls the naive algorithm makes, with garbage data. Used to
profile speed gains when offloading to DSP.

## Compiling

The ususal autotools dance:

```
autoreconf -vif
./configure
make
```

## Running

To run without offload:

```
unset OPENCV_OPENCL_DEVICE
./fakealgo
```

To run with hardware offload:

```
export OPENCV_OPENCL_DEVICE='TI AM57:ACCELERATOR:TI Multicore C66 DSP'
./fakealgo
```
