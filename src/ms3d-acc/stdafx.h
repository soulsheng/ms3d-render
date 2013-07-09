


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

#define CUDA_ENABLE		0	//	1开启cuda

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

#define RENDERMODE_VBO		0	//	1渲染模式VBO, 0渲染模式常规
#define RENDERMODE_POINT	0	//	1渲染点,  0渲染面

#define RENDERMODE_MOVING	1	//  1 动画

#define ENABLE_CONSOLE_WINDOW	1	// 1 打开另一个控制台窗口
#define ENABLE_TIMER		1		// 1 计时
#define ENABLE_FPS_COUNT	0		// 1 帧速

#define ENABLE_TIMER_VBO_MAP		0		// 1 VBO计时MAP/UNMAP

#define COUNT_MODEL			43//43		// 模型数目

#define ENABLE_DRAW_REPEAT	0	//1		重复渲染1个模型
#define ENABLE_DRAW			1	//1		渲染开启
