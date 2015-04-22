package com.piotrglazar.opencl;

import org.jocl.Pointer;

public class FloatArray {

    private final float[] data;
    private final Pointer dataPointer;

    public FloatArray(float[] data) {
        this.data = data;
        this.dataPointer = Pointer.to(data);
    }

    public float[] getData() {
        return data;
    }

    public Pointer getDataPointer() {
        return dataPointer;
    }

    public int getLength() {
        return data.length;
    }
}
