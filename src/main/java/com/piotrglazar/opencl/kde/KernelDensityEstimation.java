package com.piotrglazar.opencl.kde;

import com.google.common.base.Charsets;
import com.google.common.primitives.Floats;
import com.piotrglazar.opencl.core.OpenClMetadataFactory;
import com.piotrglazar.opencl.core.OpenClCommandQueue;
import com.piotrglazar.opencl.core.OpenClCommandWrapper;
import com.piotrglazar.opencl.core.OpenClContext;
import com.piotrglazar.opencl.core.OpenClExecutor;
import com.piotrglazar.opencl.core.OpenClMetadata;
import com.piotrglazar.opencl.core.OpenClSettings;
import com.piotrglazar.opencl.util.FloatArray;
import com.piotrglazar.opencl.util.FloatArrayReader;
import com.piotrglazar.opencl.util.FloatBuffer;
import com.piotrglazar.opencl.util.ProfilingData;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;

import static java.util.stream.Collectors.toList;

public class KernelDensityEstimation {

    public static final float X_MIN = 0;
    public static final float X_MAX = 1024;
    public static final float H = 1f;
    public static final float DENSITY = 1f;
    public static final float PI_FACTOR = (float) (1 / Math.sqrt(2 * Math.PI));

    private final FloatArray input;
    private final FloatArray output;
    private final float factor;

    public KernelDensityEstimation(FloatArray input, FloatArray output, float factor) {
        this.input = input;
        this.output = output;
        this.factor = factor;
    }

    public void executeKernels(OpenClCommandWrapper commandWrapper, OpenClExecutor executor, OpenClMetadata metadata) {
        List<KdeKernel> kdeKernels = new LinkedList<>();
        kdeKernels.add(new KdeNaiveKernel(commandWrapper, executor));

        try (OpenClContext context = getOpenClContext(commandWrapper, metadata);
             OpenClCommandQueue commandQueue = getOpenClCommandQueue(commandWrapper, metadata, context);
             FloatBuffer inputGpu = FloatBuffer.inputBuffer(commandWrapper, context, input);
             FloatBuffer outputGpu = FloatBuffer.outputBuffer(commandWrapper, context, output)) {

            KdeContext kdeContext = new KdeContext(inputGpu, outputGpu, output, X_MIN, factor, DENSITY, H,
                    input.getLength(), output.getLength());
            for (KdeKernel kdeKernel : kdeKernels) {
                ProfilingData profilingData = kdeKernel.execute(kdeContext, context, commandQueue);

                System.out.println("Total: " + (profilingData.getDuration()) / 1e6 + " ms");
                dumpArray(kdeKernel.getName(), output.getData(), false);
            }

        }
    }

    public static void main(String[] args) throws URISyntaxException, IOException {
        OpenClSettings.enableExceptions();

        FloatArray input = readInputArray();
        float factor = PI_FACTOR / input.getLength() * H;

        KernelDensityEstimation kde = new KernelDensityEstimation(input, getOutputArray(), factor);

        OpenClCommandWrapper openClCommandWrapper = new OpenClCommandWrapper();
        OpenClExecutor executor = new OpenClExecutor(openClCommandWrapper);
        OpenClMetadata metadata = new OpenClMetadataFactory(openClCommandWrapper).createMetadata();

        kde.executeKernels(openClCommandWrapper, executor, metadata);
    }

    private static FloatArray getOutputArray() {
        int outputWidth = (int) ((X_MAX - X_MIN) / DENSITY);
        return FloatArray.empty(outputWidth);
    }

    private static FloatArray readInputArray() {
        return new FloatArrayReader().read("sample/big_sample.txt");
    }

    private OpenClCommandQueue getOpenClCommandQueue(OpenClCommandWrapper openClCommandWrapper,
                                                            OpenClMetadata metadata, OpenClContext context) {
        return new OpenClCommandQueue(openClCommandWrapper, context.getContext(), metadata.getDeviceId());
    }

    private OpenClContext getOpenClContext(OpenClCommandWrapper openClCommandWrapper, OpenClMetadata metadata) {
        return new OpenClContext(openClCommandWrapper, metadata.getPlatformId(), metadata.getDeviceId());
    }

    private void dumpArray(String kernelName, float[] output, boolean shouldDump) {
        String path = "dump" + output.length + kernelName + ".txt";

        if (shouldDump) {
            List<String> outputValues = Floats.asList(output).stream().map(Object::toString).collect(toList());
            try {
                Files.write(Paths.get("C:", "tmp", path), outputValues, Charsets.UTF_8);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }


        for (int i = 0; i < 10; ++i) {
            System.out.println(output[i]);
        }
    }
}
