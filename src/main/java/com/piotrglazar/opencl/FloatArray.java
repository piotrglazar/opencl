package com.piotrglazar.opencl;

import org.jocl.Pointer;

public class FloatArray {

    private final float[] data;

    public FloatArray(float[] data) {
        this.data = data;
    }

    public float[] getData() {
        return data;
    }

    public Pointer getDataPointer() {
        return Pointer.to(data);
    }

    public int getLength() {
        return data.length;
    }

    public static FloatArray empty(int size) {
        return new FloatArray(new float[size]);
    }
}
