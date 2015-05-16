package com.piotrglazar.opencl.core;

import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

public class OpenClContext implements AutoCloseable {

    private final OpenClCommandWrapper commandWrapper;
    private final cl_context context;

    public OpenClContext(OpenClCommandWrapper commandWrapper, cl_platform_id platformId, cl_device_id deviceId) {
        this.commandWrapper = commandWrapper;
        this.context = commandWrapper.getContext(platformId, deviceId);
    }

    public cl_context getContext() {
        return context;
    }

    @Override
    public void close() {
        commandWrapper.releaseContext(context);
    }
}
