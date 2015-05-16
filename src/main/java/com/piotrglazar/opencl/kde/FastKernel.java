package com.piotrglazar.opencl.kde;

import com.piotrglazar.opencl.core.*;

public class FastKernel extends Kernel {

    private static final String NAME = "kde_fast";
    private static final int GROUP_SIZE = 1024;
    private static final int MAX_LOCAL_COPY = 4096;

    private final OpenClKernelSource kernelSource;

    public FastKernel(OpenClCommandWrapper commandWrapper, OpenClExecutor executor) {
        super(commandWrapper, executor);
        kernelSource = OpenClKernelSource.getKernelSourceCode("kernel", NAME);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected OpenClEvent execute(OpenClCommandQueue commandQueue, OpenClKernel kernel, Context context) {
        return executor.submitAndWait(commandQueue, kernel, getNumberOfGroups(context), GROUP_SIZE);
    }

    @Override
    protected OpenClKernelSource getKernelSource() {
        return kernelSource;
    }

    private int getNumberOfGroups(Context context) {
        return (context.getOutputWidth() / MAX_LOCAL_COPY) * MAX_LOCAL_COPY;
    }
}
