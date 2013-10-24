
#ifndef		_SIMPLE_OPTIMIZATION_H_
#define		_SIMPLE_OPTIMIZATION_H_

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "helper_math.h"

#define SIZE_THREAD_X		256
#define SIZE_BLOCK_X		16

#define SIZE_PER_BONE		1 //每个顶点关联骨骼的数目
#define MATRIX_SIZE_LINE		4

enum Matrix_Separate_Mode {
	NO_SEPARATE,		//	不拆分，相邻  1个float属于相邻矩阵的  1个float
	HALF_SEPARATE,		//	半拆分，相邻  4个float属于相邻矩阵的  4个float，矩阵一行
	COMPLETE_SEPARATE	//	全拆分，相邻16个float属于相邻矩阵的16个float，矩阵整体
};// 矩阵数组中相邻矩阵的存储方式



#endif