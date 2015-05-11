package com.piotrglazar.opencl;

public interface KdeKernel {

    ProfilingData execute(KdeContext kdeContext, OpenClContext context, OpenClCommandQueue commandQueue);

    String getName();
}
