package com.piotrglazar.opencl;

import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import com.google.common.io.Files;
import com.google.common.io.Resources;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.List;
import java.util.Scanner;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_WRITE_ONLY;
import static org.jocl.CL.CL_PROGRAM_BUILD_LOG;
import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetEventProfilingInfo;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clGetProgramBuildInfo;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseEvent;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;
import static org.jocl.CL.clWaitForEvents;

public class KernelDensityEstimation {

    public static final float X_MIN = 0;
    public static final float X_MAX = 256; //1024;
    public static final float H = 0.5f;
    public static final float DENSITY = 0.25f;
    public static final float PI_FACTOR = (float) (1 / Math.sqrt(2 * 3.141592653589793f));

    public static void main(String[] args) throws URISyntaxException, IOException {
        URL sample = Resources.getResource("sample/samp4");
        Scanner sampleScanner = new Scanner(new File(sample.toURI()));
        int outputWidth = (int) ((X_MAX - X_MIN) / DENSITY);

        float[] input = readInput(sampleScanner);
        float[] output = new float[outputWidth];
        float factor = PI_FACTOR / input.length;

        cl_platform_id platformId = getPlatformId();
        cl_device_id deviceId = getDeviceId(platformId);
        cl_context context = createContext(platformId, deviceId);

        /////////////////////////////////////////////////////////////////
        // Create an OpenCL command queue
        /////////////////////////////////////////////////////////////////
        cl_command_queue commandQueue = createCommandQueue(context, deviceId);

        /////////////////////////////////////////////////////////////////
        // Create OpenCL memory buffers
        /////////////////////////////////////////////////////////////////
        Pointer inputPointer = Pointer.to(input);
        cl_mem inputGpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * input.length,
                inputPointer, null);
        Pointer outputPointer = Pointer.to(output);
        cl_mem outputGpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * output.length,
                outputPointer, null);

        /////////////////////////////////////////////////////////////////
        // Load CL file, build CL program object, create CL kernel object
        /////////////////////////////////////////////////////////////////
        //const char * filename  = "kde_group_kernel.cl";//kde_simplest.cl";//"kde_vector.cl";//"kde_group_kernel.cl";//"kde_naive_kernel.cl";
//        std::cout << "using: " << kernelFile << std::endl;//filename << std::endl;
//        std::string  sourceStr = convertToString(kernelFile);
//        const char * source    = sourceStr.c_str();
//        size_t sourceSize[]    = { strlen(source) };
//
//        program = clCreateProgramWithSource(context, 1, &source, sourceSize, &status);
//        check(status, CL_SUCCESS, "Error: Loading Binary into cl_program (clCreateProgramWithBinary)\n");
        int[] errorCode = new int[1];
        cl_program program = clCreateProgramWithSource(context, 1,
                getKernelSourceCode(), null, errorCode);
        int ret = clBuildProgram(program, 0, null/*new cl_device_id[]{deviceId}*/, null, null, null);
        if (ret != CL_SUCCESS) {
            System.out.println("wrong program " + ret);
            byte[] buffer = new byte[2048];
            clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, buffer.length * Sizeof.cl_char, Pointer.to(buffer), null);
            System.out.println(new String(buffer, Charsets.UTF_8));
        }

    /* create a cl program executable for all the devices specified */
//        if (CL_SUCCESS != clBuildProgram(program, 1, devices, NULL, NULL, NULL))
//        {
//            std::cout <<  "Error: Building Program (clBuildProgram)\n";
//            size_t len;
//            char buffer[2048];
//
//            clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
//                    sizeof(buffer), buffer, &len);
//            std::cout << buffer << std::endl;
//            return 1;
//        }

    /* get a kernel object handle for a kernel with the given name */
//        kernel = clCreateKernel(program, "templateKernel", &status);
//        check(status, CL_SUCCESS, "Error: Creating Kernel from program. (clCreateKernel)\n");
//
//        return 0;

        cl_kernel kernel = clCreateKernel(program, "templateKernel", errorCode);

        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inputGpu));
//        check(clSetKernelArg(kernel, 0, sizeof(cl_mem), static_cast<void*>(&inputBuffer)),
//        CL_SUCCESS, "Error: Setting kernel argument. (input)\n");


    /* the output array to the kernel */
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outputGpu));
//        check(clSetKernelArg(kernel, 1, sizeof(cl_mem), static_cast<void *>(&outputBuffer)),
//        CL_SUCCESS, "Error: Setting kernel argument. (output)\n");


    /* xmin */
        clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[]{X_MIN}));
//        check(clSetKernelArg(kernel, 2, sizeof(cl_float), static_cast<void *>(&xmin)),
//        CL_SUCCESS, "Error: Setting kernel argument. (xmin)\n");


    /* factor */
        clSetKernelArg(kernel, 3, Sizeof.cl_float, Pointer.to(new float[]{factor}));
//        check(clSetKernelArg(kernel, 3, sizeof(cl_float), static_cast<void *>(&factor)),
//        CL_SUCCESS, "Error: Setting kernel argument. (factor)\n");


    /* den */
        clSetKernelArg(kernel, 4, Sizeof.cl_float, Pointer.to(new float[]{DENSITY}));
//        check(clSetKernelArg(kernel, 4, sizeof(cl_float), static_cast<void *>(&den)),
//        CL_SUCCESS, "Error: Setting kernel argument. (den)\n");

    /* h */
        clSetKernelArg(kernel, 5, Sizeof.cl_float, Pointer.to(new float[]{H}));
//        check(clSetKernelArg(kernel, 5, sizeof(cl_float), static_cast<void *>(&h)),
//        CL_SUCCESS, "Error: Setting kernel argument. (h)\n");

    /* input size */
        clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{input.length}));
//        check(clSetKernelArg(kernel, 6, sizeof(int), static_cast<void *>(&inputWidth)),
//        CL_SUCCESS, "Error: Setting kernel argument. (input size)\n");

    /* output size */
        clSetKernelArg(kernel, 7, Sizeof.cl_int, Pointer.to(new int[]{outputWidth}));
//        check(clSetKernelArg(kernel, 7, sizeof(int), static_cast<void *>(&outputWidth)),
//        CL_SUCCESS, "Error: Setting kernel argument. (output size)\n");



    /*
     * Enqueue a kernel run call.
     */
        cl_event[] events = new cl_event[]{new cl_event()};
        int ret2 = clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{outputWidth}, new long[]{1}, 0, events, null);
        if (ret2 != CL.CL_SUCCESS) {
            System.out.println(ret);
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
        clEnqueueReadBuffer(commandQueue, outputGpu, CL_TRUE, 0, outputWidth * Sizeof.cl_float, outputPointer, 0, null, null);
//        check(clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, outputWidth * sizeof(float), output,
//                0, NULL, NULL), CL_SUCCESS, "Error: clEnqueueReadBuffer failed. (clEnqueueReadBuffer)\n");
//
//
        dumpArray(factor, input, output);

        clReleaseMemObject(outputGpu);
        clReleaseMemObject(inputGpu);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    private static void dumpArray(float factor, float[] input, float[] output) {
        float x = X_MIN;
        float eps = 0.00001f;
        int ok = 0;
        for (int i = 0; i < 10; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < input.length; ++j) {
                sum += Math.exp(-0.5f * Math.pow(((x - input[j]) / H), 2));
                float result = Math.abs(output[i] - ((factor * sum) / H));
                if (result < eps) {
                    ++ok;
                }
                x += DENSITY;
            }
        }
        System.out.println("percentage of valid answers: " + (ok / 10));
    }

    private static String[] getKernelSourceCode() throws IOException, URISyntaxException {
        File kernelFile = new File(Resources.getResource("kernel/kde_naive.cl").toURI());
        List<String> kernelLines = Files.readLines(kernelFile, Charsets.UTF_8);
        return new String[]{Joiner.on("").join(kernelLines) };
//        return kernelLines.toArray(new String[kernelLines.size()]);
    }

    private static cl_command_queue createCommandQueue(cl_context context, cl_device_id deviceId) {
        return clCreateCommandQueue(context, deviceId, 0, null);
    }

    private static cl_platform_id getPlatformId() {
        int platformIndex = 0;

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        return platforms[platformIndex];
    }

    private static cl_device_id getDeviceId(cl_platform_id platform) {
        int deviceIndex = 0;
        long deviceType = CL.CL_DEVICE_TYPE_GPU;

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        return devices[deviceIndex];
    }

    private static cl_context createContext(cl_platform_id platform, cl_device_id device) {
        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        return clCreateContext(contextProperties, 1, new cl_device_id[]{device},
                null, null, null);
    }

    private static float[] readInput(Scanner sampleScanner) {
        int size = sampleScanner.nextInt();
        float[] result = new float[size];

        for (int i = 0; i < size; ++i) {
            String raw = sampleScanner.next();
            result[i] = Float.parseFloat(raw);
        }

        return result;
    }
}
