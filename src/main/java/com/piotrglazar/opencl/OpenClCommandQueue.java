package com.piotrglazar.opencl;

import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;

public class OpenClCommandQueue implements AutoCloseable {

    private final OpenClCommandWrapper commandWrapper;
    private final cl_command_queue commandQueue;

    public OpenClCommandQueue(OpenClCommandWrapper commandWrapper, cl_context context, cl_device_id deviceId) {
        this.commandWrapper = commandWrapper;
        this.commandQueue = commandWrapper.getCommandQueue(context, deviceId);
    }

    public cl_command_queue getCommandQueue() {
        return commandQueue;
    }

    @Override
    public void close() {
        commandWrapper.releaseCommandQueue(commandQueue);
    }
}
