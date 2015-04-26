package com.piotrglazar.opencl;

import org.jocl.cl_kernel;

public class OpenClKernel implements AutoCloseable {

    private final OpenClCommandWrapper commandWrapper;
    private final cl_kernel kernel;
    private final String name;

    public OpenClKernel(OpenClCommandWrapper commandWrapper, OpenClProgram program, String kernelName) {
        this.commandWrapper = commandWrapper;
        this.kernel = commandWrapper.createKernel(program.getProgram(), kernelName);
        this.name = kernelName;
    }

    public void addKernelArgument(int argumentNumber, FloatBuffer buffer) {
        commandWrapper.addKernelArgument(kernel, argumentNumber, buffer);
    }

    public void addKernelArgument(int argumentNumber, float value) {
        commandWrapper.addKernelArgument(kernel, argumentNumber, value);
    }

    public void addKernelArgument(int argumentNumber, int value) {
        commandWrapper.addKernelArgument(kernel, argumentNumber, value);
    }

    public cl_kernel getKernel() {
        return kernel;
    }

    public String getName() {
        return name;
    }

    @Override
    public void close() {
        commandWrapper.releaseKernel(kernel);
    }
}
