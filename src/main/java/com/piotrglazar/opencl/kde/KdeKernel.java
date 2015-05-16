package com.piotrglazar.opencl.kde;

import com.piotrglazar.opencl.core.OpenClCommandQueue;
import com.piotrglazar.opencl.core.OpenClContext;
import com.piotrglazar.opencl.util.ProfilingData;

public interface KdeKernel {

    ProfilingData execute(KdeContext kdeContext, OpenClContext context, OpenClCommandQueue commandQueue);

    String getName();
}
