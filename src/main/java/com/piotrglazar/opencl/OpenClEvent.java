package com.piotrglazar.opencl;

import org.jocl.cl_event;

public class OpenClEvent {

    private final cl_event event;

    public OpenClEvent(cl_event event) {
        this.event = event;
    }

    public cl_event getEvent() {
        return event;
    }
}
