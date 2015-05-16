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

public class KdeNaiveKernel implements KdeKernel {

    private static final String NAME = "kde_naive";

    private final OpenClKernelSource kernelSource;
    private final OpenClCommandWrapper openClCommandWrapper;
    private final OpenClExecutor executor;

    public KdeNaiveKernel(OpenClCommandWrapper openClCommandWrapper, OpenClExecutor executor) {
        this.openClCommandWrapper = openClCommandWrapper;
        this.executor = executor;
        this.kernelSource = OpenClKernelSource.getKernelSourceCode("kernel", NAME);
    }

    @Override
    public ProfilingData execute(KdeContext kdeContext, OpenClContext context, OpenClCommandQueue commandQueue) {
        try (OpenClProgram program = new OpenClProgram(openClCommandWrapper, context, kernelSource);
             OpenClKernel kernel = new OpenClKernel(openClCommandWrapper, program, "kernelDensityEstimation")) {

            kernel.addKernelArgument(0, kdeContext.getInputGpu());
            kernel.addKernelArgument(1, kdeContext.getOutputGpu());
            kernel.addKernelArgument(2, kdeContext.getxMin());
            kernel.addKernelArgument(3, kdeContext.getFactor());
            kernel.addKernelArgument(4, kdeContext.getDensity());
            kernel.addKernelArgument(5, kdeContext.getH());
            kernel.addKernelArgument(6, kdeContext.getInputWidth());
            kernel.addKernelArgument(7, kdeContext.getOutputWidth());

            OpenClEvent event = executor.submitAndWait(commandQueue, kernel, kdeContext.getOutputWidth());
            ProfilingData profilingData = executor.getProfilingData(event);

            executor.releaseEvent(event);
            executor.copyFromGpuToMemory(commandQueue, kdeContext.getOutputGpu(), kdeContext.getOutput());

            return profilingData;
        }
    }

    @Override
    public String getName() {
        return "kde_naive";
    }
}
