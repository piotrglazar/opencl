package com.piotrglazar.opencl;

import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;

public class OpenClCommandWrapper {

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

    public cl_command_queue getCommandQueue(cl_context context, cl_device_id deviceId) {
        int[] errorCode = new int[1];

        cl_command_queue commandQueue = clCreateCommandQueue(context, deviceId, 0, errorCode);
        verifyCallSucceeded(errorCode[0], "clCreateCommandQueue");

        return commandQueue;
    }

    public void releaseCommandQueue(cl_command_queue commandQueue) {
        verifyCallSucceeded(clReleaseCommandQueue(commandQueue), "clReleaseCommandQueue");
    }
}
