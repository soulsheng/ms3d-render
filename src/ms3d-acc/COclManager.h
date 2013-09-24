#pragma once

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include "MilkshapeModel.h"	


typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetGLContextInfoKHR_fn)(
	const cl_context_properties *properties,
	cl_gl_context_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret);

// Rename references to this dynamically linked function to avoid
// collision with static link version
#define clGetGLContextInfoKHR clGetGLContextInfoKHR_proc
static clGetGLContextInfoKHR_fn clGetGLContextInfoKHR;

class COclManager
{
public:

	COclManager();
	~COclManager();
	bool Setup_OpenCL( const char *program_source , const char *kernel_name );
	void initialize();
	void Cleanup();
protected:

private:
	// OpenCL specific
	cl_context	g_context;
	cl_command_queue g_cmd_queue;
	cl_program	g_program;
	cl_kernel	g_kernel;

	cl_uint     g_min_align;
	cl_device_id g_device_ID;

public:
	MilkshapeModel* m_model;

};