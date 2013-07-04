


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
#define RENDERMODE_POINT	1	//	1��Ⱦ��,  0��Ⱦ��

#define RENDERMODE_MOVING	1	//  1 ����
