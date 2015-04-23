package com.piotrglazar.opencl;

import org.jocl.cl_event;

public class OpenClExecutor {

    private final OpenClCommandWrapper commandWrapper;

    public OpenClExecutor(OpenClCommandWrapper commandWrapper) {
        this.commandWrapper = commandWrapper;
    }

    public cl_event submitAndWait(OpenClCommandQueue commandQueue, OpenClKernel kernel, int globalThreads) {
        cl_event event = commandWrapper.enqueue(commandQueue.getCommandQueue(), kernel.getKernel(), globalThreads);
        commandWrapper.waitForEvents(event);

        return event;
    }
}
