package com.piotrglazar.opencl;

public interface KdeKernel {

    void execute(KdeContext kdeContext, OpenClContext context);
}
