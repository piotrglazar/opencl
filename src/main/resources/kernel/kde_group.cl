__kernel void templateKernel(const __global float *input, __global  float *output, const float xmin, const float factor,
                                const float den, const float h, const int inputSize, const int outputSize)
{
	int MAX_LOCAL_COPY = 1536;
	int GROUP_SIZE = 1024;
	int gid = get_group_id(0);
	__local float cache[1536];
	int offset = gid * MAX_LOCAL_COPY, howMany;
	event_t cp;
	float x = xmin + den * get_global_id(0);
    float tmp = 0.0f;
    int rem = 0;

	for (int k = 0; k < inputSize / MAX_LOCAL_COPY; ++k)
	{
		cp = async_work_group_copy(cache, input + offset, MAX_LOCAL_COPY, 0);
		wait_group_events(1, &cp);

        for (int i = 0; i < MAX_LOCAL_COPY; ++i)
        	tmp += exp(-0.5f * pow(((x - cache[i]) / h), 2));
        
		offset = (offset + MAX_LOCAL_COPY) % inputSize;
	}

    if ((rem = inputSize % MAX_LOCAL_COPY) != 0)
    {
        cp = async_work_group_copy(cache, input + offset, rem, 0);
		wait_group_events(1, &cp);

        for (int i = 0; i < rem; ++i)
        	tmp += exp(-0.5f * pow(((x - cache[i]) / h), 2));
    }


	tmp *= (factor / h);

    barrier(CLK_LOCAL_MEM_FENCE);

	cache[get_local_id(0)] = tmp;
    if ((gid + 1 == get_num_groups(0)) && ((rem = outputSize % GROUP_SIZE) != 0))
        cp = async_work_group_copy(output + GROUP_SIZE * gid, cache, rem, 0);
    else
    	cp = async_work_group_copy(output + GROUP_SIZE * gid, cache, GROUP_SIZE, 0);
    wait_group_events(1, &cp);
}
