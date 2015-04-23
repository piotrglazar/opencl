package com.piotrglazar.opencl;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_event;
import org.jocl.cl_kernel;

import java.io.IOException;
import java.net.URISyntaxException;

import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clGetEventProfilingInfo;
import static org.jocl.CL.clReleaseEvent;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clSetKernelArg;
import static org.jocl.CL.clWaitForEvents;

public class KernelDensityEstimation {

    public static final float X_MIN = 0;
    public static final float X_MAX = 256; //1024;
    public static final float H = 0.5f;
    public static final float DENSITY = 0.25f;
    public static final float PI_FACTOR = (float) (1 / Math.sqrt(2 * 3.141592653589793f));

    public static void main(String[] args) throws URISyntaxException, IOException {
        int outputWidth = (int) ((X_MAX - X_MIN) / DENSITY);

        OpenClKernelSource kernel = OpenClKernelSource.getKernelSourceCode("kernel/kde_naive.cl");
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
                OpenClProgram program = new OpenClProgram(openClCommandWrapper, context, kernel)) {

            int[] errorCode = new int[1];

    /* get a kernel object handle for a kernel with the given name */
//        kernel = clCreateKernel(program, "templateKernel", &status);
//        check(status, CL_SUCCESS, "Error: Creating Kernel from program. (clCreateKernel)\n");
//
//        return 0;

            cl_kernel clKernel = clCreateKernel(program.getProgram(), "templateKernel", errorCode);

            clSetKernelArg(clKernel, 0, Sizeof.cl_mem, Pointer.to(inputGpu.getMemoryBuffer()));
//        check(clSetKernelArg(kernel, 0, sizeof(cl_mem), static_cast<void*>(&inputBuffer)),
//        CL_SUCCESS, "Error: Setting kernel argument. (input)\n");


    /* the output array to the kernel */
            clSetKernelArg(clKernel, 1, Sizeof.cl_mem, Pointer.to(outputGpu.getMemoryBuffer()));
//        check(clSetKernelArg(kernel, 1, sizeof(cl_mem), static_cast<void *>(&outputBuffer)),
//        CL_SUCCESS, "Error: Setting kernel argument. (output)\n");


    /* xmin */
            clSetKernelArg(clKernel, 2, Sizeof.cl_float, Pointer.to(new float[]{X_MIN}));
//        check(clSetKernelArg(kernel, 2, sizeof(cl_float), static_cast<void *>(&xmin)),
//        CL_SUCCESS, "Error: Setting kernel argument. (xmin)\n");


    /* factor */
            clSetKernelArg(clKernel, 3, Sizeof.cl_float, Pointer.to(new float[]{factor}));
//        check(clSetKernelArg(kernel, 3, sizeof(cl_float), static_cast<void *>(&factor)),
//        CL_SUCCESS, "Error: Setting kernel argument. (factor)\n");


    /* den */
            clSetKernelArg(clKernel, 4, Sizeof.cl_float, Pointer.to(new float[]{DENSITY}));
//        check(clSetKernelArg(kernel, 4, sizeof(cl_float), static_cast<void *>(&den)),
//        CL_SUCCESS, "Error: Setting kernel argument. (den)\n");

    /* h */
            clSetKernelArg(clKernel, 5, Sizeof.cl_float, Pointer.to(new float[]{H}));
//        check(clSetKernelArg(kernel, 5, sizeof(cl_float), static_cast<void *>(&h)),
//        CL_SUCCESS, "Error: Setting kernel argument. (h)\n");

    /* input size */
            clSetKernelArg(clKernel, 6, Sizeof.cl_int, Pointer.to(new int[]{input.getLength()}));
//        check(clSetKernelArg(kernel, 6, sizeof(int), static_cast<void *>(&inputWidth)),
//        CL_SUCCESS, "Error: Setting kernel argument. (input size)\n");

    /* output size */
            clSetKernelArg(clKernel, 7, Sizeof.cl_int, Pointer.to(new int[]{outputWidth}));
//        check(clSetKernelArg(kernel, 7, sizeof(int), static_cast<void *>(&outputWidth)),
//        CL_SUCCESS, "Error: Setting kernel argument. (output size)\n");



    /*
     * Enqueue a kernel run call.
     */
            cl_event[] events = new cl_event[]{new cl_event()};
            int ret2 = clEnqueueNDRangeKernel(commandQueue.getCommandQueue(), clKernel, 1, null, new long[]{outputWidth}, new long[]{1}, 0, events, null);
            if (ret2 != CL.CL_SUCCESS) {
                System.out.println(ret2);
            }
//        check(clEnqueueNDRangeKernel(commandQueue, kernel, 1, /* number of dimensions */ NULL, globalThreads,
//                localThreads, 0, NULL, &events[0]), CL_SUCCESS, "Error: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)\n");



    /* wait for the kernel call to finish execution */
            clWaitForEvents(1, events);
//        check(clWaitForEvents(1, &events[0]), CL_SUCCESS, "Error: Waiting for kernel run to finish. (clWaitForEvents)\n");


    /* profilling info */
            long[] start = new long[1];
            long[] stop = new long[1];
            clGetEventProfilingInfo(events[0], CL.CL_PROFILING_COMMAND_START, Sizeof.cl_ulong, Pointer.to(start), null);
            clGetEventProfilingInfo(events[0], CL.CL_PROFILING_COMMAND_END, Sizeof.cl_ulong, Pointer.to(stop), null);
            System.out.println(start[0]);
            System.out.println(stop[0]);
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
            clReleaseEvent(events[0]);

    /* Enqueue readBuffer*/
            clEnqueueReadBuffer(commandQueue.getCommandQueue(), outputGpu.getMemoryBuffer(), CL_TRUE, 0, outputWidth * Sizeof.cl_float, output.getDataPointer(), 0, null, null);
//        check(clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, outputWidth * sizeof(float), output,
//                0, NULL, NULL), CL_SUCCESS, "Error: clEnqueueReadBuffer failed. (clEnqueueReadBuffer)\n");
//
//
            dumpArray(factor, input.getData(), output.getData());

            clReleaseKernel(clKernel);
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
