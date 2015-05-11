package com.piotrglazar.opencl;

public class KdeNaiveKernel implements KdeKernel {

    private final OpenClKernelSource kernelSource;
    private final OpenClCommandWrapper openClCommandWrapper;

    public KdeNaiveKernel(OpenClCommandWrapper openClCommandWrapper) {
        this.openClCommandWrapper = openClCommandWrapper;
        this.kernelSource = OpenClKernelSource.getKernelSourceCode("kernel", "kde_naive");
    }

    @Override
    public void execute(KdeContext kdeContext, OpenClContext context) {
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
        }
    }
}
