package com.piotrglazar.opencl;

public class KdeContext {

    private final FloatBuffer inputGpu;
    private final FloatBuffer outputGpu;
    private final float xMin;
    private final float factor;
    private final float density;
    private final float h;
    private final int inputWidth;
    private final int outputWidth;

    public KdeContext(FloatBuffer inputGpu, FloatBuffer outputGpu, float xMin, float factor, float density, float h,
                      int inputWidth, int outputWidth) {
        this.inputGpu = inputGpu;
        this.outputGpu = outputGpu;
        this.xMin = xMin;
        this.factor = factor;
        this.density = density;
        this.h = h;
        this.inputWidth = inputWidth;
        this.outputWidth = outputWidth;
    }

    public FloatBuffer getInputGpu() {
        return inputGpu;
    }

    public FloatBuffer getOutputGpu() {
        return outputGpu;
    }

    public float getxMin() {
        return xMin;
    }

    public float getFactor() {
        return factor;
    }

    public float getDensity() {
        return density;
    }

    public float getH() {
        return h;
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public int getOutputWidth() {
        return outputWidth;
    }
}
