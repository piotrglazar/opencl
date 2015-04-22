package com.piotrglazar.opencl;

import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;

public class OpenClCommandWrapper {

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
}
