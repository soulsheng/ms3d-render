
#ifndef		_SIMPLE_OPTIMIZATION_H_
#define		_SIMPLE_OPTIMIZATION_H_

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "helper_math.h"

#define SIZE_THREAD_X		256
#define SIZE_BLOCK_X		16

#define SIZE_PER_BONE		1 //ÿ�����������������Ŀ
#define MATRIX_SIZE_LINE		4

enum Matrix_Separate_Mode {
	NO_SEPARATE,		//	����֣�����  1��float�������ھ����  1��float
	HALF_SEPARATE,		//	���֣�����  4��float�������ھ����  4��float������һ��
	COMPLETE_SEPARATE	//	ȫ��֣�����16��float�������ھ����16��float����������
};// �������������ھ���Ĵ洢��ʽ



#endif