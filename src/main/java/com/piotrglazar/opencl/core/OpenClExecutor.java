package com.piotrglazar.opencl.core;

import com.piotrglazar.opencl.util.FloatArray;
import com.piotrglazar.opencl.util.FloatBuffer;
import com.piotrglazar.opencl.util.ProfilingData;
import org.jocl.cl_event;

public class OpenClExecutor {

    private final OpenClCommandWrapper commandWrapper;

    public OpenClExecutor(OpenClCommandWrapper commandWrapper) {
        this.commandWrapper = commandWrapper;
    }

    public OpenClEvent submitAndWait(OpenClCommandQueue commandQueue, OpenClKernel kernel, int globalThreads,
                                     int localThreads) {
        cl_event event = commandWrapper.enqueue(commandQueue.getCommandQueue(), kernel.getKernel(), globalThreads,
                localThreads);
        commandWrapper.waitForEvents(event);

        return new OpenClEvent(event);
    }

    public ProfilingData getProfilingData(OpenClEvent event) {
        long start = commandWrapper.getEventStart(event.getEvent());
        long end = commandWrapper.getEventEnd(event.getEvent());

        return new ProfilingData(start, end);
    }

    public void releaseEvent(OpenClEvent event) {
        commandWrapper.releaseEvent(event.getEvent());
    }

    public void copyFromGpuToMemory(OpenClCommandQueue commandQueue, FloatBuffer outputGpu, FloatArray output) {
        commandWrapper.copyFromGpuToMemory(commandQueue.getCommandQueue(), outputGpu.getMemoryBuffer(),
                output.getLength(), output.getDataPointer());
    }
}
