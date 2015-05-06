package com.piotrglazar.opencl;

import org.jocl.CL;

public final class OpenClSettings {

    private OpenClSettings() {

    }

    public static void enableExceptions() {
        CL.setExceptionsEnabled(true);
    }
}
