package com.piotrglazar.opencl.core;

import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

public class OpenClMetadata {

    private final cl_platform_id platformId;
    private final cl_device_id deviceId;

    public OpenClMetadata(cl_platform_id platformId, cl_device_id deviceId) {
        this.platformId = platformId;
        this.deviceId = deviceId;
    }

    public cl_platform_id getPlatformId() {
        return platformId;
    }

    public cl_device_id getDeviceId() {
        return deviceId;
    }
}
