
#ifndef		_SIMPLE_OPTIMIZATION_H_
#define		_SIMPLE_OPTIMIZATION_H_

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "helper_math.h"

#define SIZE_THREAD_X		256
#define SIZE_BLOCK_X		64
#define SIZE_BLOCK_STATIC	0

#define SIZE_PER_BONE		1 //ÿ�����������������Ŀ
#define MATRIX_SIZE_LINE		4

#define ENABLE_MEMORY_ALIGN		0 // ����

enum Matrix_Separate_Mode {
	NO_SEPARATE,		//	����֣�����  1��float�������ھ����  1��float
	HALF_SEPARATE,		//	���֣�����  4��float�������ھ����  4��float������һ��
	COMPLETE_SEPARATE	//	ȫ��֣�����16��float�������ھ����16��float����������
};// �������������ھ���Ĵ洢��ʽ


struct Vector4 { float x,y,z,w; };
struct Vector1 { float x; };
struct Vector1i { int x; };

#if ENABLE_MEMORY_ALIGN
typedef float4		FLOAT4;
typedef float1		FLOAT1;
typedef int1		INT1;
#else
typedef Vector4		FLOAT4;
typedef Vector1		FLOAT1;
typedef Vector1i	INT1;
#endif

static __inline__ __host__ __device__ Vector4 make_vector4(float x, float y, float z, float w)
{
  Vector4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ float4 make_float4(float1 x, float1 y, float1 z, float1 w)
{
  float4 t; t.x = x.x; t.y = y.x; t.z = z.x; t.w = w.x; return t;
}

inline __host__ __device__ Vector4 operator+(Vector4 a, Vector4 b)
{
	Vector4 t;
	t.x = a.x + b.x; t.y = a.y + b.y; t.z = a.z + b.z; t.w = a.w + b.w; 
	return t;
}
inline __host__ __device__ void operator+=(Vector4 &a, Vector4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ void operator*=(Vector4 &a, Vector4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ Vector4 operator*(Vector4 a, Vector4 b)
{
    Vector4 t;
	t.x = a.x * b.x; t.y = a.y * b.y; t.z = a.z * b.z; t.w = a.w * b.w; 
	return t;
}

#endif