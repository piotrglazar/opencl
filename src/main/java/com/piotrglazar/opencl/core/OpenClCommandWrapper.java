package com.piotrglazar.opencl.core;

import com.piotrglazar.opencl.util.FloatBuffer;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_WRITE_ONLY;
import static org.jocl.CL.CL_QUEUE_PROFILING_ENABLE;
import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetEventProfilingInfo;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseEvent;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;
import static org.jocl.CL.clWaitForEvents;

public class OpenClCommandWrapper {

    public void releaseMemoryBuffer(cl_mem memoryBuffer) {
        verifyCallSucceeded(clReleaseMemObject(memoryBuffer), "clReleaseMemObject");
    }

    public cl_mem createOutputBuffer(cl_context context, int length, Pointer pointer) {
        return createBuffer(context, length, pointer, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR);
    }

    public cl_mem createInputBuffer(cl_context context, int length, Pointer pointer) {
        return createBuffer(context, length, pointer, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    }

    private cl_mem createBuffer(cl_context context, int length, Pointer pointer, long flags) {
        int[] errorCode = new int[1];
        cl_mem mem = clCreateBuffer(context, flags, Sizeof.cl_float * length, pointer, errorCode);

        verifyCallSucceeded(errorCode[0], "clCreateBuffer");

        return mem;
    }

    public void releaseContext(cl_context context) {
        verifyCallSucceeded(clReleaseContext(context), "clReleaseContext");
    }

    public cl_context getContext(cl_platform_id platformId, cl_device_id deviceId) {
        int[] errorCode = new int[1];

        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platformId);

        cl_context context = clCreateContext(contextProperties, 1, new cl_device_id[]{deviceId}, null, null, errorCode);
        verifyCallSucceeded(errorCode[0], "clCreateContext");

        return context;
    }

    public cl_device_id getDeviceId(cl_platform_id platformId, long deviceType, int deviceNumber) {
        int numberOfDevices = getNumberOfDevices(platformId, deviceType);
        cl_device_id devices[] = new cl_device_id[numberOfDevices];

        verifyCallSucceeded(clGetDeviceIDs(platformId, deviceType, numberOfDevices, devices, null), "clGetDeviceIDs");

        return devices[deviceNumber];
    }

    public int getNumberOfDevices(cl_platform_id platformId, long deviceType) {
        int[] numberOfDevices = new int[1];

        verifyCallSucceeded(clGetDeviceIDs(platformId, deviceType, 0, null, numberOfDevices), "clGetDeviceIDs");

        return numberOfDevices[0];
    }

    public cl_platform_id getPlatformId(int platformNumber) {
        cl_platform_id platforms[] = new cl_platform_id[getNumberOfPlatforms()];

        verifyCallSucceeded(clGetPlatformIDs(platforms.length, platforms, null), "clGetPlatformIDs");

        return platforms[platformNumber];
    }

    public int getNumberOfPlatforms() {
        int[] numberOfPlatforms = new int[1];

        verifyCallSucceeded(clGetPlatformIDs(0, null, numberOfPlatforms), "clGetPlatformIDs");

        return numberOfPlatforms[0];
    }

    private void verifyCallSucceeded(int result, String functionName) {
        if (result != CL_SUCCESS) {
            throw new OpenClApiException(functionName, result);
        }
    }

    public cl_command_queue createCommandQueue(cl_context context, cl_device_id deviceId) {
        int[] errorCode = new int[1];

        cl_command_queue commandQueue = clCreateCommandQueue(context, deviceId, CL_QUEUE_PROFILING_ENABLE, errorCode);
        verifyCallSucceeded(errorCode[0], "clCreateCommandQueue");

        return commandQueue;
    }

    public void releaseCommandQueue(cl_command_queue commandQueue) {
        verifyCallSucceeded(clReleaseCommandQueue(commandQueue), "clReleaseCommandQueue");
    }

    public cl_program createAndBuildProgram(cl_context context, String[] kernelSourceCode) {
        int[] errorCode = new int[1];
        cl_program program = clCreateProgramWithSource(context, 1, kernelSourceCode, null, errorCode);
        verifyCallSucceeded(errorCode[0], "clCreateProgramWithSource");

        // TODO: build fail info
        errorCode[0] = clBuildProgram(program, 0, null, null, null, null);
        verifyCallSucceeded(errorCode[0], "clBuildProgram");

        return program;
    }

    public void releaseProgram(cl_program program) {
        verifyCallSucceeded(clReleaseProgram(program), "clReleaseProgram");
    }

    public cl_kernel createKernel(cl_program program, String kernelName) {
        int[] errorCode = new int[1];
        cl_kernel kernel = clCreateKernel(program, kernelName, errorCode);

        verifyCallSucceeded(errorCode[0], "clCreateKernel");

        return kernel;
    }

    public void releaseKernel(cl_kernel kernel) {
        verifyCallSucceeded(clReleaseKernel(kernel), "clReleaseKernel");
    }

    public void addKernelArgument(cl_kernel kernel, int argumentNumber, FloatBuffer buffer) {
        verifyCallSucceeded(clSetKernelArg(kernel, argumentNumber, Sizeof.cl_mem,
                Pointer.to(buffer.getMemoryBuffer())), "clSetKernelArg");
    }

    public void addKernelArgument(cl_kernel kernel, int argumentNumber, float value) {
        verifyCallSucceeded(clSetKernelArg(kernel, argumentNumber, Sizeof.cl_float,
                Pointer.to(new float[]{ value })), "clSetKernelArg");
    }

    public void addKernelArgument(cl_kernel kernel, int argumentNumber, int value) {
        verifyCallSucceeded(clSetKernelArg(kernel, argumentNumber, Sizeof.cl_int,
                Pointer.to(new int[]{value})), "clSetKernelArg");
    }

    public cl_event enqueue(cl_command_queue commandQueue, cl_kernel kernel, int globalThreads, int localThreads) {
        cl_event event = new cl_event();
        verifyCallSucceeded(clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{globalThreads},
                new long[]{localThreads}, 0, null, event), "clEnqueueNDRangeKernel");
        return event;
    }

    public void waitForEvents(cl_event... events) {
        verifyCallSucceeded(clWaitForEvents(events.length, events), "clWaitForEvents");
    }

    public long getEventStart(cl_event event) {
        return getEventData(event, CL.CL_PROFILING_COMMAND_START);
    }

    public long getEventEnd(cl_event event) {
        return getEventData(event, CL.CL_PROFILING_COMMAND_END);
    }

    private long getEventData(cl_event event, int type) {
        long[] value = new long[1];
        verifyCallSucceeded(clGetEventProfilingInfo(event, type, Sizeof.cl_ulong,
                Pointer.to(value), null), "clGetEventProfilingInfo");

        return value[0];
    }

    public void releaseEvent(cl_event event) {
        verifyCallSucceeded(clReleaseEvent(event), "clReleaseEvent");
    }

    public void copyFromGpuToMemory(cl_command_queue commandQueue, cl_mem gpuBuffer, int howMany, Pointer output) {
        verifyCallSucceeded(clEnqueueReadBuffer(commandQueue, gpuBuffer, CL_TRUE, 0, howMany * Sizeof.cl_float,
                output, 0, null, null), "clEnqueueReadBuffer");
    }
}
