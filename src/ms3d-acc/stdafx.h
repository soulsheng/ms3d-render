


#include <windows.h>												// Header File For Windows
#include <stdio.h>													// Header File For Standard Input/Output

#include <time.h>
#include <conio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
using namespace std;

// OpenGL
#include "glew/glew.h"
#include <gl\gl.h>													// Header File For The OpenGL32 Library
#include <gl\glu.h>													// Header File For The GLu32 Library
#include <gl\glaux.h>												// Header File For The Glaux Library

#pragma comment( lib, "opengl32.lib" )								// Search For OpenGL32.lib While Linking ( NEW )
#pragma comment( lib, "glu32.lib" )									// Search For GLu32.lib While Linking    ( NEW )
#pragma comment( lib, "glaux.lib" )									// Search For GLaux.lib While Linking    ( NEW )
#pragma comment( lib, "glew32.lib" )									// Search For GLaux.lib While Linking    ( NEW )

#define CUDA_ENABLE		0	//	1����cuda

#if CUDA_ENABLE
// cuda

#include <cuda/cuda_gl_interop.h>
#include <cuda/common/cutil_inline.h>

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")
#ifdef _DEBUG
#pragma comment(lib, "cutil32D.lib")
#else
#pragma comment(lib, "cutil32.lib")
#endif
#endif

#define RENDERMODE_VBO		0	//	1��ȾģʽVBO, 0��Ⱦģʽ����
#define RENDERMODE_POINT	0	//	1��Ⱦ��,  0��Ⱦ��

#define RENDERMODE_MOVING	1	//  1 ����

#define ENABLE_CONSOLE_WINDOW	1	// 1 ����һ������̨����
#define ENABLE_TIMER		0		// 1 ��ʱ
#define ENABLE_FPS_COUNT	0		// 1 ֡��

#define ENABLE_TIMER_VBO_MAP		0		// 1 VBO��ʱMAP/UNMAP

#define COUNT_MODEL						1		// ģ����Ŀ
#define COUNT_MODEL_SIMULATE		4				// ģ���ģ�ͣ�43����1M������  1012134������

#define ENABLE_DRAW_REPEAT	0	//1		�ظ���Ⱦ1��ģ��
#define ENABLE_DRAW			1	//1		��Ⱦ����

#define ENABLE_OPTIMIZE	1 //�Ż�

#define ENABLE_OPENMP	0 //OpenMP

#define ENABLE_CROSSARRAY	0 //��������

#define SIZE_PER_BONE		1 //ÿ�����������������Ŀ
#define ENABLE_OPTIMIZE_SSE	0 //SSE

#if ENABLE_CROSSARRAY
#define ELEMENT_COUNT_POINT		8
#define STRIDE_POINT			5
#else
#define ELEMENT_COUNT_POINT		4
#define STRIDE_POINT			0
#endif

#define MATRIX_SIZE_LINE		4

#define ELEMENT_COUNT_MATIRX	(MATRIX_SIZE_LINE*4)

#define ENABLE_OPENCL	1	//OpenCL ����
#define SIZE_OPENCL_TEST	(1<<16)	// ����64K����, MESH[1]

#define NAME_STRING_PLATFORM	"AMD Accelerated Parallel Processing"
#define NAME_STRING_PLATFORM_2	"NVIDIA CUDA"
#define NAME_STRING_PLATFORM_3	"Intel(R) OpenCL"

#define  LocalWorkX		8
#define  LocalWorkY		8
#define  LocalWorkSizeDef	1//�ֲ��߳̿����Ĭ����Ŀ

#define ENABLE_MESH_SINGLE	0//����mesh����������

#define ENABLE_MESH_MIX		1//�ϲ�mesh

#define ENABLE_CL_GL_INTER	0//OpenCl OpenGL ������

#define TIME_CL_MEMERY_READ		1// ����OpenCL�ڴ����ݴ���ʱ�䣬��
#define TIME_CL_MEMERY_WRITE	1// ����OpenCL�ڴ����ݴ���ʱ�䣬д
#define TIME_CL_MEMERY_CALCULATE	1// ����OpenCL�ڴ����ݼ���ʱ��