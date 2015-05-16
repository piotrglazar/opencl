package com.piotrglazar.opencl.util;

import com.piotrglazar.opencl.core.OpenClCommandWrapper;
import com.piotrglazar.opencl.core.OpenClContext;
import org.jocl.cl_mem;

public class FloatBuffer implements AutoCloseable {

    private final OpenClCommandWrapper commandWrapper;
    private final cl_mem memoryBuffer;

    private FloatBuffer(OpenClCommandWrapper commandWrapper, cl_mem memoryBuffer) {
        this.commandWrapper = commandWrapper;
        this.memoryBuffer = memoryBuffer;
    }

    public static FloatBuffer inputBuffer(OpenClCommandWrapper commandWrapper, OpenClContext context, FloatArray floatArray) {
        cl_mem mem = commandWrapper.createInputBuffer(context.getContext(), floatArray.getLength(), floatArray.getDataPointer());
        return new FloatBuffer(commandWrapper, mem);
    }

    public static FloatBuffer outputBuffer(OpenClCommandWrapper commandWrapper, OpenClContext context, FloatArray floatArray) {
        cl_mem mem = commandWrapper.createOutputBuffer(context.getContext(), floatArray.getLength(), floatArray.getDataPointer());
        return new FloatBuffer(commandWrapper, mem);
    }

    public cl_mem getMemoryBuffer() {
        return memoryBuffer;
    }

    @Override
    public void close() {
        commandWrapper.releaseMemoryBuffer(memoryBuffer);
    }
}
