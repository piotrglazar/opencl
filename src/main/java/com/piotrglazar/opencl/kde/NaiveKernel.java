package com.piotrglazar.opencl.kde;

import com.piotrglazar.opencl.core.OpenClCommandQueue;
import com.piotrglazar.opencl.core.OpenClCommandWrapper;
import com.piotrglazar.opencl.core.OpenClEvent;
import com.piotrglazar.opencl.core.OpenClExecutor;
import com.piotrglazar.opencl.core.OpenClKernel;
import com.piotrglazar.opencl.core.OpenClKernelSource;

public class NaiveKernel extends Kernel {

    private static final int LOCAL_THREAD_COUNT = 1;
    private static final String NAME = "kde_naive";

    private final OpenClKernelSource kernelSource;

    public NaiveKernel(OpenClCommandWrapper openClCommandWrapper, OpenClExecutor executor) {
        super(openClCommandWrapper, executor);
        this.kernelSource = OpenClKernelSource.getKernelSourceCode("kernel", NAME);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected OpenClEvent execute(OpenClCommandQueue commandQueue, OpenClKernel kernel, Context context) {
        return executor.submitAndWait(commandQueue, kernel, context.getOutputWidth(), LOCAL_THREAD_COUNT);
    }

    @Override
    protected OpenClKernelSource getKernelSource() {
        return kernelSource;
    }
}
