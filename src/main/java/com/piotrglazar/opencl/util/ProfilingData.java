package com.piotrglazar.opencl.util;

public class ProfilingData {

    private final long start;
    private final long stop;

    public ProfilingData(long start, long stop) {
        this.start = start;
        this.stop = stop;
    }

    public long getStart() {
        return start;
    }

    public long getStop() {
        return stop;
    }

    public long getDurationMillis() {
        return (long) ((stop - start) / 1e6);
    }
}
