package com.piotrglazar.opencl.core;

public class OpenClApiException extends RuntimeException {

    public OpenClApiException(String functionName, int errorCode) {
        super("Failed to call " + functionName + ", errorCode " + errorCode);
    }
}
