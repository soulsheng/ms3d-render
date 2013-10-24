
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "helper_math.h"

#include "SimpleOptimizations.cuh"

__global__ void
transformVectorByMatrix4( const  float4 *pInput, const int *pIndex, float4 *pMatrix, float4 *pOutput,  int sizeMax,  const float *pWeight)
{
	//size_t index = get_global_id(0) + get_global_id(1) *get_global_size(0);
	size_t index = threadIdx.x + threadIdx.y*blockDim.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x * blockDim.y;

	
	if( index >= sizeMax )
		return;

	int offset = pIndex[index*SIZE_PER_BONE+0]*4 ;
	float weight = pWeight[index*SIZE_PER_BONE+0] ;
	float4 weight4 = make_float4( weight,weight,weight,weight ) ;

	float4 m0 = pMatrix[offset+0] * weight4 ;
	float4 m1 = pMatrix[offset+1] * weight4 ;
	float4 m2 = pMatrix[offset+2] * weight4 ;
	float4 m3 = pMatrix[offset+3] * weight4 ;

	for(int i=1;i<SIZE_PER_BONE; i++)
	{
		offset = pIndex[index*SIZE_PER_BONE+i]*4 ;
		weight = pWeight[index*SIZE_PER_BONE+i] ;
		weight4 = make_float4( weight, weight, weight, weight ) ;

		m0 += pMatrix[offset+0] * weight4 ;
		m1 += pMatrix[offset+1] * weight4 ;
		m2 += pMatrix[offset+2] * weight4 ;
		m3 += pMatrix[offset+3] * weight4 ;
	}

	float4 pIn = pInput[index];
	float4 px = make_float4(pIn.x, pIn.x, pIn.x, pIn.x) ;
	float4 py = make_float4(pIn.y, pIn.y, pIn.y, pIn.y) ;
	float4 pz = make_float4(pIn.z, pIn.z, pIn.z, pIn.z) ;

	pOutput[index] = px * m0 + py * m1 + pz * m2 + m3;

}


__global__ void
transformVectorByMatrix4One( const float4 *pInput, const int *pIndex, float4 *pMatrix, float4 *pOutput,  int sizeMax,  const float *pWeight)
{
	//size_t index = get_global_id(0) + get_global_id(1) *get_global_size(0);
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

	if( indexBase >= sizeMax )
		return;

	int index=indexBase;
#if SIZE_BLOCK_STATIC
		for( ; index<sizeMax; index+=blockDim.x * gridDim.x )
#endif
		{

			float4 pIn = pInput[index];
			float4 px = make_float4(pIn.x, pIn.x, pIn.x, pIn.x) ;
			float4 py = make_float4(pIn.y, pIn.y, pIn.y, pIn.y) ;
			float4 pz = make_float4(pIn.z, pIn.z, pIn.z, pIn.z) ;

			int offset = pIndex[index]*4 ;

			float4 m0 = pMatrix[offset+0] ;
			float4 m1 = pMatrix[offset+1] ;
			float4 m2 = pMatrix[offset+2] ;
			float4 m3 = pMatrix[offset+3] ;

			pOutput[index] = px * m0 + py * m1 + pz * m2 + m3 ;
		}
}


/* 坐标矩阵变换
pVertex  : 坐标
pMatrix : 矩阵
*/
template<typename F4>
__device__ void transformVec3ByMatrix4(F4* pVertexIn, float1 pMatrix[], F4* pVertexOut)
{
	F4 vertexIn = *pVertexIn;
	F4 vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0].x + vertexIn.y * pMatrix[1].x + vertexIn.z * pMatrix[2].x + pMatrix[3].x ; 
	vertexOut.y = vertexIn.x * pMatrix[1*4+0].x + vertexIn.y * pMatrix[1*4+1].x + vertexIn.z * pMatrix[1*4+2].x + pMatrix[1*4+3].x  ; 
	vertexOut.z = vertexIn.x * pMatrix[2*4+0].x + vertexIn.y * pMatrix[2*4+1].x + vertexIn.z * pMatrix[2*4+2].x + pMatrix[2*4+3].x  ;
	*pVertexOut = vertexOut;
}
template<typename F4>
__device__ void transformVec3ByMatrix4(F4* pVertexIn, F4 pMatrix[], F4* pVertexOut)
{
	F4 vertexIn = *pVertexIn;
	F4 vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0].x + vertexIn.y * pMatrix[0].y + vertexIn.z * pMatrix[0].z + pMatrix[0].w ; 
	vertexOut.y = vertexIn.x * pMatrix[1].x + vertexIn.y * pMatrix[1].y + vertexIn.z * pMatrix[1].z + pMatrix[1].w  ; 
	vertexOut.z = vertexIn.x * pMatrix[2].x + vertexIn.y * pMatrix[2].y + vertexIn.z * pMatrix[2].z + pMatrix[2].w  ;
	*pVertexOut = vertexOut;
}

template<typename F4>
__device__ void transformVec3ByMatrix4_f4(F4* pVertexIn, float4 pMatrix[], F4* pVertexOut)
{
	F4 vertexIn = *pVertexIn;
	F4 vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0].x + vertexIn.y * pMatrix[0].y + vertexIn.z * pMatrix[0].z + pMatrix[0].w ; 
	vertexOut.y = vertexIn.x * pMatrix[1].x + vertexIn.y * pMatrix[1].y + vertexIn.z * pMatrix[1].z + pMatrix[1].w  ; 
	vertexOut.z = vertexIn.x * pMatrix[2].x + vertexIn.y * pMatrix[2].y + vertexIn.z * pMatrix[2].z + pMatrix[2].w  ;
	*pVertexOut = vertexOut;
}

	// 按矩阵索引
template<typename F4>
__device__ void indexByFloat44( F4* pBuffer , F4* pMat , int index )
	{
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			pMat[j] = pBuffer[index * MATRIX_SIZE_LINE + j];
		}
	}


__global__ void updateVectorByMatrix(float4* pVertexIn, int size, float1* pMatrix, float4* pVertexOut)
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

		for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){

		float4   matrix[MATRIX_SIZE_LINE];

		// 读取操作数：初始的顶点坐标
		float4   vertexIn = pVertexIn[i];

		// 读取操作数：顶点对应的矩阵
		int      matrixIndex = int(vertexIn.w + 0.5);// float to int

		
		indexByFloat44( (float4*)pMatrix, matrix, matrixIndex );

		// 执行操作：对坐标执行矩阵变换，得到新坐标
		transformVec3ByMatrix4( &vertexIn, matrix, pVertexOut+i);
	}//for
}

extern "C" bool
runCUDADevice( const float *pInput, const int *pIndex, float *pMatrix, float *pOutput,  int sizeMax,  const float *pWeight )
{
	int nCountThreadsPerBlock = SIZE_THREAD_X;
    dim3 block( nCountThreadsPerBlock, 1, 1);

#if SIZE_BLOCK_STATIC
	dim3 grid( SIZE_BLOCK_X, 1, 1);
#else
	int nCountBlocks = (sizeMax + nCountThreadsPerBlock - 1) / nCountThreadsPerBlock ;
	dim3 grid( nCountBlocks, 1, 1);
#endif

    // execute the kernel
    transformVectorByMatrix4One<<< grid, block >>>( (float4*)pInput, pIndex, (float4*)pMatrix, (float4*)pOutput, sizeMax, pWeight );
	//updateVectorByMatrix<<< grid, block >>>( (float4*)pInput, sizeMax, (float1*)pMatrix, (float4*)pOutput );

	return true;
}