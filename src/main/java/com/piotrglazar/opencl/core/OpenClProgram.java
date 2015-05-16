package com.piotrglazar.opencl.core;

import org.jocl.cl_program;

public class OpenClProgram implements AutoCloseable {

    private final OpenClCommandWrapper commandWrapper;
    private final cl_program program;


    public OpenClProgram(OpenClCommandWrapper commandWrapper, OpenClContext context, OpenClKernelSource kernel) {
        this.commandWrapper = commandWrapper;
        this.program = commandWrapper.createAndBuildProgram(context.getContext(), kernel.getKernelSourceCode());
    }

    public cl_program getProgram() {
        return program;
    }

    @Override
    public void close() {
        commandWrapper.releaseProgram(program);
    }
}
