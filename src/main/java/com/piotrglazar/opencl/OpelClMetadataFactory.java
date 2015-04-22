package com.piotrglazar.opencl;

import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

import static org.jocl.CL.CL_DEVICE_TYPE_GPU;

public class OpelClMetadataFactory {

    private static final long DEVICE_TYPE = CL_DEVICE_TYPE_GPU;

    private final OpenClCommandWrapper openClCommandWrapper;

    public OpelClMetadataFactory(OpenClCommandWrapper openClCommandWrapper) {
        this.openClCommandWrapper = openClCommandWrapper;
    }

    public OpenClMetadata createMetadata() {
        cl_platform_id platformId = getPlatformId();
        cl_device_id deviceId = getDeviceId(platformId);

        return new OpenClMetadata(platformId, deviceId);
    }

    private cl_device_id getDeviceId(cl_platform_id platform) {
        return openClCommandWrapper.getDeviceId(platform, DEVICE_TYPE, 0);
    }

    private cl_platform_id getPlatformId() {
        return openClCommandWrapper.getPlatformId(0);
    }
}
