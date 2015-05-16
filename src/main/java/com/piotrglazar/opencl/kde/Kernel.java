package com.piotrglazar.opencl.kde;

import com.piotrglazar.opencl.core.OpenClCommandQueue;
import com.piotrglazar.opencl.core.OpenClCommandWrapper;
import com.piotrglazar.opencl.core.OpenClContext;
import com.piotrglazar.opencl.core.OpenClEvent;
import com.piotrglazar.opencl.core.OpenClExecutor;
import com.piotrglazar.opencl.core.OpenClKernel;
import com.piotrglazar.opencl.core.OpenClKernelSource;
import com.piotrglazar.opencl.core.OpenClProgram;
import com.piotrglazar.opencl.util.ProfilingData;

public abstract class Kernel {

    protected final OpenClCommandWrapper commandWrapper;
    protected final OpenClExecutor executor;

    protected Kernel(OpenClCommandWrapper commandWrapper, OpenClExecutor executor) {
        this.commandWrapper = commandWrapper;
        this.executor = executor;
    }

    public ProfilingData execute(Context kdeContext, OpenClContext context, OpenClCommandQueue commandQueue) {
        try (OpenClProgram program = new OpenClProgram(commandWrapper, context, getKernelSource());
             OpenClKernel kernel = new OpenClKernel(commandWrapper, program, "kernelDensityEstimation")) {

            kernel.addKernelArgument(0, kdeContext.getInputGpu());
            kernel.addKernelArgument(1, kdeContext.getOutputGpu());
            kernel.addKernelArgument(2, kdeContext.getxMin());
            kernel.addKernelArgument(3, kdeContext.getFactor());
            kernel.addKernelArgument(4, kdeContext.getDensity());
            kernel.addKernelArgument(5, kdeContext.getH());
            kernel.addKernelArgument(6, kdeContext.getInputWidth());
            kernel.addKernelArgument(7, kdeContext.getOutputWidth());

            OpenClEvent event = execute(commandQueue, kernel, kdeContext);
            ProfilingData profilingData = executor.getProfilingData(event);

            executor.releaseEvent(event);
            executor.copyFromGpuToMemory(commandQueue, kdeContext.getOutputGpu(), kdeContext.getOutput());

            return profilingData;
        }
    }

    public abstract String getName();

    protected abstract OpenClEvent execute(OpenClCommandQueue commandQueue, OpenClKernel kernel, Context context);

    protected abstract OpenClKernelSource getKernelSource();
}
