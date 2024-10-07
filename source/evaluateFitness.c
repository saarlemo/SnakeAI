#define CL_TARGET_OPENCL_VERSION 300
#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Input validation
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:invalidNumInputs",
                          "One input matrix is required");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:invalidNumOutputs",
                          "One output is required.");
    }
    if (!mxIsSingle(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:inputNotSingle",
                          "Input matrix must be of type single.");
    }
    if (mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:inputComplex",
                          "Input matrix must be real.");
    }

    // Get input matrices dimensions
    mwSize numWeights = mxGetM(prhs[0]);
    mwSize numGenomes = mxGetN(prhs[0]);
    
    // Get pointers to input data
    float *weights = (float *)mxGetData(prhs[0]);

    // Create output matrix
    mwSize mC = numGenomes;
    plhs[0] = mxCreateNumericMatrix(mC, 1, mxSINGLE_CLASS, mxREAL);
    float *C = (float *)mxGetData(plhs[0]);

    // OpenCL initialization
    cl_int err;
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;

    // Get OpenCL platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS) {
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:clGetPlatformIDs",
                          "Failed to get OpenCL platforms.");
    }
    cl_platform_id *platforms = (cl_platform_id *)mxMalloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);

    // Use the first platform
    platform = platforms[0];

    // Get OpenCL device
    cl_device_id device = NULL;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        // Try CPU device if GPU is not available
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            mexErrMsgIdAndTxt("MATLAB:evaluateFitness:clGetDeviceIDs",
                              "Failed to get OpenCL device.");
        }
    }

    // Create context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // Create OpenCL buffers
    //printf("Creating OpenCL buffers ... ");
    size_t size1 = numGenomes * numWeights * sizeof(float); // First argument size
    size_t size2 = numGenomes * sizeof(float); // Second argument size

    cl_mem buffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY, size1, NULL, &err);
    cl_mem buffer2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size2, NULL, &err);
    //printf("Done\n");

    // Write data to buffers
    err = clEnqueueWriteBuffer(queue, buffer1, CL_TRUE, 0, size1, weights, 0, NULL, NULL);

    // Read the kernel source code from the file
    //printf("Reading kernel source code ... ");
    const char *kernelFileName = "source/snake_kernel.cl";
    FILE *fp = fopen(kernelFileName, "r");
    if (!fp)
    {
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:KernelFileNotFound",
                        "Failed to load kernel file '%s'.", kernelFileName);
    }
    fseek(fp, 0, SEEK_END);
    size_t programSize = ftell(fp);
    rewind(fp);
    char *programSource = (char *)mxMalloc(programSize + 1);
    programSource[programSize] = '\0';
    fread(programSource, sizeof(char), programSize, fp);
    fclose(fp);
    //printf("Done\n");

    // Build and compile the kernel
    //printf("Building and compiling the kernel ... ");
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        // Print build log in case of errors
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)mxMalloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:clBuildProgram",
                          "Failed to build program:\n%s", log);
    }
    cl_kernel kernel = clCreateKernel(program, "snake_kernel", &err);
    //printf("Done\n");

    // Set kernel arguments
    int numWeightsInt = (int)numWeights;
    int numGenomesInt = (int)numGenomes;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer1);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer2);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &numGenomesInt);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &numWeightsInt);

    // Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &numGenomes, NULL, 0, NULL, NULL);
    clFinish(queue);

    // Read back the result
    err = clEnqueueReadBuffer(queue, buffer2, CL_TRUE, 0, size2, C, 0, NULL, NULL);

    // Release OpenCL resources
    clReleaseMemObject(buffer1);
    clReleaseMemObject(buffer2);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    mxFree(platforms);
}