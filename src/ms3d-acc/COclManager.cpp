#include "stdafx.h"
#include "COclManager.h"

#include "common\utils.h"


COclManager::COclManager()
{

}

COclManager::~COclManager()
{

}

bool COclManager::Setup_OpenCL( const char *program_source , const char *kernel_name)
{
	cl_device_id devices[16];
	size_t cb;
	cl_uint size_ret = 0;
	cl_int err;
	int num_cores;
	char device_name[128] = {0};

	static const char buildOpts[] = "-cl-fast-relaxed-math";

	cl_platform_id intel_platform_id = GetIntelOCLPlatform();
	if( intel_platform_id == NULL )
	{
		printf("ERROR: Failed to find Intel OpenCL platform.\n");
		return false;
	}

	cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)intel_platform_id, NULL };

	// create the OpenCL context on a CPU/PG 
	g_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL);
	
	if (g_context == (cl_context)0)
		return false;

	// get the list of CPU devices associated with context
	err = clGetContextInfo(g_context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
	clGetContextInfo(g_context, CL_CONTEXT_DEVICES, cb, devices, NULL);
	g_cmd_queue = clCreateCommandQueue(g_context, devices[0], 0, NULL);
	if (g_cmd_queue == (cl_command_queue)0)
	{
		Cleanup();
		return false;
	}

	char *sources = ReadSources(program_source);	//read program .cl source file
	g_program = clCreateProgramWithSource(g_context, 1, (const char**)&sources, NULL, NULL);
	if (g_program == (cl_program)0)
	{
		printf("ERROR: Failed to create Program with source...\n");
		Cleanup();
		free(sources);
		return false;
	}

	err = clBuildProgram(g_program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("ERROR: Failed to build program...\n");
		Cleanup();
		free(sources);
		return false;
	}

	g_kernel = clCreateKernel(g_program, kernel_name, NULL);

	if (g_kernel == (cl_kernel)0)
	{
		printf("ERROR: Failed to create kernel...\n");
		Cleanup();
		free(sources);
		return false;
	}
	
	free(sources);

	// use first device ID
	g_device_ID = devices[0];
	err = clGetDeviceInfo(g_device_ID, CL_DEVICE_NAME, 128, device_name, NULL);
	if (err!=CL_SUCCESS)
	{
		printf("ERROR: Failed to get device information (device name)...\n");
		Cleanup();
		return false;
	}
	printf("Using device %s...\n", device_name);

	err = clGetDeviceInfo(g_device_ID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cores, NULL);
	if (err!=CL_SUCCESS)
	{
		printf("ERROR: Failed to get device information (max compute units)...\n");
		Cleanup();
		return false;
	}
	printf("Using %d compute units...\n", num_cores);


	err = clGetDeviceInfo(g_device_ID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &g_min_align, NULL);
	if (err!=CL_SUCCESS)
	{
		printf("ERROR: Failed to get device information (max memory base address align size)...\n");
		Cleanup();
		return false;
	}
	g_min_align /= 8; //in bytes
	printf("Buffer alignment required for zero-copying is %d bytes (CL_DEVICE_MEM_BASE_ADDR_ALIGN)\n\n", g_min_align);

	return true; // success...
}

void COclManager::Cleanup()
{
	if( g_kernel ) {clReleaseKernel( g_kernel );  g_kernel = NULL;}
	if( g_program ) {clReleaseProgram( g_program );  g_program = NULL;}
	if( g_cmd_queue ) {clReleaseCommandQueue( g_cmd_queue );  g_cmd_queue = NULL;}
	if( g_context ) {clReleaseContext( g_context );  g_context = NULL;}
	//host memory
	//    if(g_pfInput) {_aligned_free( g_pfInput ); g_pfInput = NULL;}
	//    if(g_pfRegularOutput) {_aligned_free( g_pfRegularOutput ); g_pfRegularOutput = NULL;}
	//if(g_pfOCLOutput) {_aligned_free( g_pfOCLOutput ); g_pfOCLOutput = NULL;}
	//unInitialize();
}

void COclManager::initialize()
{
	g_context = NULL;
	g_cmd_queue = NULL;
	g_program = NULL;
	g_kernel = NULL;

	g_min_align = 0;
	g_device_ID =0;
}