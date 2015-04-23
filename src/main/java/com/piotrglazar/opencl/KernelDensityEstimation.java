package com.piotrglazar.opencl;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_event;

import java.io.IOException;
import java.net.URISyntaxException;

import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clGetEventProfilingInfo;
import static org.jocl.CL.clReleaseEvent;
import static org.jocl.CL.clWaitForEvents;

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

    /*
     * Enqueue a kernel run call.
     */
            cl_event event = new cl_event();
            int ret2 = clEnqueueNDRangeKernel(commandQueue.getCommandQueue(), kernel.getKernel(), 1, null, new long[]{outputWidth}, new long[]{1}, 0, null, event);
            if (ret2 != CL.CL_SUCCESS) {
                System.out.println(ret2);
            }
//        check(clEnqueueNDRangeKernel(commandQueue, kernel, 1, /* number of dimensions */ NULL, globalThreads,
//                localThreads, 0, NULL, &events[0]), CL_SUCCESS, "Error: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)\n");



    /* wait for the kernel call to finish execution */
            clWaitForEvents(1, new cl_event[]{event});
//        check(clWaitForEvents(1, &events[0]), CL_SUCCESS, "Error: Waiting for kernel run to finish. (clWaitForEvents)\n");


    /* profilling info */
            long[] start = new long[1];
            long[] stop = new long[1];
            clGetEventProfilingInfo(event, CL.CL_PROFILING_COMMAND_START, Sizeof.cl_ulong, Pointer.to(start), null);
            clGetEventProfilingInfo(event, CL.CL_PROFILING_COMMAND_END, Sizeof.cl_ulong, Pointer.to(stop), null);
            System.out.println("" + (stop[0] - start[0]) / 1e6 + " ms");
//        cl_ulong startTime, finishTime;
//
//        check(clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL),
//                CL_SUCCESS, "Error: eventProfillingInfo: startTime");
//        check(clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &finishTime, NULL),
//                CL_SUCCESS, "Error: eventProfillingInfo: finishTime");
//
//        std::cout << "Start time: "<< startTime << " finishTime " << finishTime << " execution time [ms] "
//                << (finishTime - startTime) / 1000000 << std::endl;
//
//        check(clReleaseEvent(events[0]), CL_SUCCESS, "Error: Release event object. (clReleaseEvent)\n");
            clReleaseEvent(event);

    /* Enqueue readBuffer*/
            clEnqueueReadBuffer(commandQueue.getCommandQueue(), outputGpu.getMemoryBuffer(), CL_TRUE, 0, outputWidth * Sizeof.cl_float, output.getDataPointer(), 0, null, null);
//        check(clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, outputWidth * sizeof(float), output,
//                0, NULL, NULL), CL_SUCCESS, "Error: clEnqueueReadBuffer failed. (clEnqueueReadBuffer)\n");
//
//
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
