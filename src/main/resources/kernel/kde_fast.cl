__kernel void kernelDensityEstimation(const __global float *input, __global  float *output, const float xmin, const float factor,
                                const float den, const float h, const int inputSize, const int outputSize) {
    int MAX_LOCAL_COPY = 4096;
    int GROUP_SIZE = 1024;
    int gid = get_group_id(0);
    __local float cache[4096];
    int offset = gid * MAX_LOCAL_COPY;
    event_t cp;
    float4 xs = (float4) (xmin + 4 * den * get_global_id(0));
    xs.S1 = xs.S0 + den;
    xs.S2 = xs.S1 + den;
    xs.S3 = xs.S2 + den;
    float4 tmp = (float4) (0.0f);
    int4 toPower = (int4) (2);
    int rem = 0;
    int copySize = 0;

    for (int k = 0; k < (inputSize + MAX_LOCAL_COPY - 1) / MAX_LOCAL_COPY; ++k) {
        copySize = (offset + MAX_LOCAL_COPY < inputSize) ? MAX_LOCAL_COPY : inputSize - offset;
        cp = async_work_group_copy(cache, input + offset, copySize, 0);
        wait_group_events(1, &cp);

        for (int i = 0; i < copySize; ++i) {
            tmp += native_exp(-0.5f * pown(((float4) (cache[i]) - xs) / h, toPower));
        }
        
        offset = (offset + copySize) % inputSize;
    }
    
    tmp *= (factor / h);

    barrier(CLK_LOCAL_MEM_FENCE);

    vstore4(tmp, get_local_id(0), cache);
    if ((gid + 1 == get_num_groups(0)) && ((rem = outputSize % (GROUP_SIZE * 4)) != 0)) {
        cp = async_work_group_copy(output + 4 * GROUP_SIZE * gid, cache, rem, 0);
    } else {
        cp = async_work_group_copy(output + 4 * GROUP_SIZE * gid, cache, 4 * GROUP_SIZE, 0);
    }
    wait_group_events(1, &cp);
}
