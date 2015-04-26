package com.piotrglazar.opencl;

import java.io.IOException;
import java.net.URISyntaxException;

public class KernelDensityEstimation {

    public static final float X_MIN = 0;
    public static final float X_MAX = 256; //1024;
    public static final float H = 0.5f;
    public static final float DENSITY = 0.25f;
    public static final float PI_FACTOR = (float) (1 / Math.sqrt(2 * 3.141592653589793f));

    public static void main(String[] args) throws URISyntaxException, IOException {
        int outputWidth = (int) ((X_MAX - X_MIN) / DENSITY);

        OpenClKernelSource kernelSource = OpenClKernelSource.getKernelSourceCode("kernel/kde_naive.cl");
        FloatArray input = new FloatArrayReader().read("sample/big_sample.txt");
        FloatArray output = FloatArray.empty(outputWidth);
        float factor = PI_FACTOR / input.getLength();

        OpenClCommandWrapper openClCommandWrapper = new OpenClCommandWrapper();
        OpelClMetadataFactory metadataFactory = new OpelClMetadataFactory(openClCommandWrapper);
        OpenClMetadata metadata = metadataFactory.createMetadata();

        try (OpenClContext context = getOpenClContext(openClCommandWrapper, metadata);
                OpenClCommandQueue commandQueue = getOpenClCommandQueue(openClCommandWrapper, metadata, context);
                FloatBuffer inputGpu = FloatBuffer.inputBuffer(openClCommandWrapper, context, input);
                FloatBuffer outputGpu = FloatBuffer.outputBuffer(openClCommandWrapper, context, output);
                OpenClProgram program = new OpenClProgram(openClCommandWrapper, context, kernelSource);
                OpenClKernel kernel = new OpenClKernel(openClCommandWrapper, program, "kernelDensityEstimation")) {


            kernel.addKernelArgument(0, inputGpu);
            kernel.addKernelArgument(1, outputGpu);
            kernel.addKernelArgument(2, X_MIN);
            kernel.addKernelArgument(3, factor);
            kernel.addKernelArgument(4, DENSITY);
            kernel.addKernelArgument(5, H);
            kernel.addKernelArgument(6, input.getLength());
            kernel.addKernelArgument(7, outputWidth);

            OpenClExecutor executor = new OpenClExecutor(openClCommandWrapper);

            OpenClEvent event = executor.submitAndWait(commandQueue, kernel, outputWidth);

            ProfilingData profilingData = executor.getProfilingData(event);
            System.out.println("Total: " + (profilingData.getDuration()) / 1e6 + " ms");

            executor.releaseEvent(event);
            executor.copyFromGpuToMemory(commandQueue, outputGpu, output);

            dumpArray(factor, input.getData(), output.getData());
        }
    }

    private static OpenClCommandQueue getOpenClCommandQueue(OpenClCommandWrapper openClCommandWrapper,
                                                            OpenClMetadata metadata, OpenClContext context) {
        return new OpenClCommandQueue(openClCommandWrapper, context.getContext(), metadata.getDeviceId());
    }

    private static OpenClContext getOpenClContext(OpenClCommandWrapper openClCommandWrapper, OpenClMetadata metadata) {
        return new OpenClContext(openClCommandWrapper, metadata.getPlatformId(), metadata.getDeviceId());
    }

    private static void dumpArray(float factor, float[] input, float[] output) {
        float x = X_MIN;
        float eps = 0.00001f;
        int ok = 0;
        for (int i = 0; i < 10; ++i) {
//            float sum = 0.0f;
//            for (int j = 0; j < input.length; ++j) {
//                sum += Math.exp(-0.5f * Math.pow(((x - input[j]) / H), 2));
//                float result = Math.abs(output[i] - ((factor * sum) / H));
//                if (result < eps) {
//                    ++ok;
//                }
//                x += DENSITY;
//            }
            System.out.println(output[i]);
        }
//        System.out.println("percentage of valid answers: " + (ok / 10));
    }
}
