#include "stdafx.h"
#include "COclManager.h"

#include "common\utils.h"

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define FILENAME_MS3D "data/tortoise.ms3d"


COclManager::COclManager()
{
	m_model = new MilkshapeModel[COUNT_MODEL];

}

COclManager::~COclManager()
{

	if ( m_model )
	{
		delete[] m_model;
		m_model = NULL;
	}
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

#if !ENABLE_CL_GL_INTER
	cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)intel_platform_id, NULL };

	// create the OpenCL context on a CPU/PG 
	g_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL);
#else

	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext(), //获得OpenGL上下文
		CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC(), //获得OpenGl设备信息
		CL_CONTEXT_PLATFORM, (cl_context_properties) intel_platform_id,  //获得平台信息
		0};

		if (!clGetGLContextInfoKHR) 
		{
			clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn) clGetExtensionFunctionAddressForPlatform(intel_platform_id, "clGetGLContextInfoKHR");
			if (!clGetGLContextInfoKHR) 
			{
				std::cout << "Failed to query proc address for clGetGLContextInfoKHR";
				return false;
			}
		}
#if 0
		size_t deviceSize = 0;
		cl_int status = clGetGLContextInfoKHR( properties, 
			CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
			0, 
			NULL, 
			&deviceSize);

		if (CL_SUCCESS != status || 0 == deviceSize )
			deviceSize=0;
#endif
		// 获取支持GL-CL互操作的OpenCL设备的ID
		cl_int status = clGetGLContextInfoKHR( properties, 
			CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
			sizeof(cl_device_id), 
			&g_device_ID, 
			NULL);

		//利用刚刚获取的设备ID创建上下文
		g_context = clCreateContext(properties, 1, &g_device_ID, NULL, NULL, &err);
#endif

		if (g_context == (cl_context)0 )
			return false;

		// get the list of CPU devices associated with context
		err = clGetContextInfo(g_context, CL_CONTEXT_DEVICES, 0, NULL, &cb);


#if !ENABLE_CL_GL_INTER
		clGetContextInfo(g_context, CL_CONTEXT_DEVICES, cb, devices, NULL);
		g_device_ID = devices[0];
#endif

		g_cmd_queue = clCreateCommandQueue(g_context, g_device_ID, 0, NULL);
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
	//g_device_ID = devices[0];
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


	// load data
	for (int i=0;i<COUNT_MODEL;i++)
	{
		m_model[i].loadModelData(FILENAME_MS3D);
		m_model[i].reloadTextures();										// Loads Model Textures
#if ENABLE_OPENCL_CPU
		m_model[i].SetupKernel( g_context, g_device_ID, g_kernel, g_cmd_queue );
#endif

#if ENABLE_NVIDIA_CUDA
		m_model[i].initializeCUDA();
#endif
	}


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

	// 释放 cuda 设备，清理相应的资源
	cudaDeviceReset();

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

bool COclManager::Setup_CUDA( int argc, char *argv[] )
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaDevice(argc, (const char **)argv);

	return true;
}
