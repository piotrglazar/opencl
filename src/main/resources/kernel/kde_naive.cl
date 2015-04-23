__kernel void kernelDensityEstimation(const __global float *input, __global  float *output, const float xmin,
                const float factor, const float den, const float h, const int inputSize, const int outputSize)
{
    int id = get_global_id(0);
    float x = (xmin + den * id);
    float tmp = 0.0f;

    for (int i = 0; i < inputSize; ++i)
        tmp += exp(-0.5f * pow(((x - input[i]) / h), 2));
    tmp *= factor / h;

    if (id < outputSize)
        output[id] = tmp;
}
