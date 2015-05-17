package com.piotrglazar.opencl.kde;

public interface OpenClApi {

    void asyncWorkGroupCopy(int offset, int n);

    void compute();
}
