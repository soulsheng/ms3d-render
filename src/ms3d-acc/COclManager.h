#pragma once

#include "CL\cl.h"
#include "MilkshapeModel.h"	

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