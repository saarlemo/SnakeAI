#define CL_TARGET_OPENCL_VERSION 300
#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#define DEBUG 0

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2) { // Input validation
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:invalidNumInputs",
                          "Two inputs are required");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:invalidNumOutputs",
                          "One output is required.");
    }
    if (!mxIsSingle(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:inputNotSingle",
                          "Input weight matrix must be of type single.");
    }
    if (mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:evaluateFitness:inputComplex",
                          "Input weight matrix must be real.");
    }

    if (!mxIsStruct(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:notStruct",
                          "Second input must be a struct.");
    }

    /* Get the number of fields in the struct */
    int numFields = mxGetNumberOfFields(prhs[1]);
    double optionsData[numFields];
    for (int i = 0; i < numFields; i++) {
        mxArray *fieldValue = mxGetFieldByNumber(prhs[1], 0, i);
        double *data = mxGetPr(fieldValue);
        optionsData[i] = (double)data[0];
    }

    #if DEBUG==1
    mexPrintf("The struct has %d fields.\n", numFields);
    /* Loop over each field and print its name and value */
    for (int i = 0; i < numFields; i++) {
        const char *fieldName = mxGetFieldNameByNumber(prhs[1], i);
        mexPrintf("Field %d: %s\n", i + 1, fieldName);

        /* Get the field value as an mxArray* */
        mxArray *fieldValue = mxGetFieldByNumber(prhs[1], 0, i);
        if (mxIsDouble(fieldValue)) {
            double *data = mxGetPr(fieldValue);
            size_t numElements = mxGetNumberOfElements(fieldValue);
            mexPrintf("Data: ");
            for (size_t j = 0; j < numElements; j++) {
                mexPrintf("%f ", data[j]);
            }
            mexPrintf("\n");
        } else {
            mexPrintf("Field is not of type double.\n");
        }
    }
    #endif
    
    // Compiler options
    size_t optionsBufferSize = 1024;
    char* compilerOptions = (char*)malloc(optionsBufferSize);
    snprintf(compilerOptions, optionsBufferSize, "-D INPUT_SIZE=%u -D HIDDEN_SIZE=%u -D OUTPUT_SIZE=%u -D N_HIDDEN=%u -D GRID_WIDTH=%u -D GRID_HEIGHT=%u -D MAX_STEPS=%u -D BONUS_STEPS=%u", (int)optionsData[0], (int)optionsData[1], (int)optionsData[2], (int)optionsData[3], (int)optionsData[4], (int)optionsData[5], (int)optionsData[6], (int)optionsData[7]);

    // Get input matrix dimensions
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
#if DEBUG==1
    printOpenCLDeviceInfo(device);
#endif
    // Create context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    const cl_queue_properties properties[] = { 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, &err);

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
    size_t bytesRead = fread(programSource, sizeof(char), programSize, fp);
    fclose(fp);
    //printf("Done\n");

    // Build and compile the kernel
    //printf("Building and compiling the kernel ... ");
    const char *source = programSource;
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    err = clBuildProgram(program, 1, &device, compilerOptions, NULL, NULL);
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

void printOpenCLDeviceInfo(cl_device_id device) {
    char buffer[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
    printf("Device: %s\n", buffer);
}