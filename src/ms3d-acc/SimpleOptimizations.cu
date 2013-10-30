
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "helper_math.h"

#include "SimpleOptimizations.cuh"

#if ENABLE_MEMORY_CONST
__constant__		float4		shared_pMatrix_f4[ MATRIX_FIX_LENGTH * MATRIX_SIZE_LINE ];// 100个矩阵空间，6.4k
#endif

__global__ void
transformVectorByMatrix4Shared( const  float4 *pInput, const int *pIndex, float4 *pMatrix, float4 *pOutput,  int sizeMax,  const float *pWeight, int sizeJoint )
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

	if( indexBase >= sizeMax )
		return;

#if ENABLE_MEMORY_SHARED
	__shared__		float4		shared_pMatrix_f4[ MATRIX_FIX_LENGTH * MATRIX_SIZE_LINE ];// 100个矩阵空间，6.4k
	if( threadIdx.x < sizeJoint )
	{
		for(int i=0;i<MATRIX_SIZE_LINE;i++)
			shared_pMatrix_f4[ threadIdx.x + i*sizeJoint ] = pMatrix[ threadIdx.x + i*sizeJoint ];	
	}
	__syncthreads();
#endif

	int index=indexBase;
#if SIZE_BLOCK_STATIC
		for( ; index<sizeMax; index+=blockDim.x * gridDim.x )
#endif
		{
#if ENABLE_MEMORY_COALESCED
			int offset = pIndex[index]*4 ;
			float weight = pWeight[index] ;
#else
			int offset = pIndex[index*SIZE_PER_BONE+0]*4 ;
			float weight = pWeight[index*SIZE_PER_BONE+0] ;
#endif//#if ENABLE_MEMORY_COALESCED 合并访问

			float4 weight4 = make_float4( weight,weight,weight,weight ) ;

			float4 m0 = shared_pMatrix_f4[offset+0] * weight4 ;
			float4 m1 = shared_pMatrix_f4[offset+1] * weight4 ;
			float4 m2 = shared_pMatrix_f4[offset+2] * weight4 ;
			float4 m3 = shared_pMatrix_f4[offset+3] * weight4 ;

			
			for(int i=1;i<SIZE_PER_BONE; i++)
			{
#if ENABLE_MEMORY_COALESCED
				offset = pIndex[index+i*sizeMax]*4 ;
				weight = pWeight[index+i*sizeMax] ;
#else
				offset = pIndex[index*SIZE_PER_BONE+i]*4 ;
				weight = pWeight[index*SIZE_PER_BONE+i] ;
#endif//#if ENABLE_MEMORY_COALESCED 合并访问

				weight4 = make_float4( weight, weight, weight, weight ) ;

				m0 += shared_pMatrix_f4[offset+0] * weight4 ;
				m1 += shared_pMatrix_f4[offset+1] * weight4 ;
				m2 += shared_pMatrix_f4[offset+2] * weight4 ;
				m3 += shared_pMatrix_f4[offset+3] * weight4 ;
			}

			float4 pIn = pInput[index];
			float4 px = make_float4(pIn.x, pIn.x, pIn.x, pIn.x) ;
			float4 py = make_float4(pIn.y, pIn.y, pIn.y, pIn.y) ;
			float4 pz = make_float4(pIn.z, pIn.z, pIn.z, pIn.z) ;

			pOutput[index] = px * m0 + py * m1 + pz * m2 + m3;
		}
}
__global__ void
transformVectorByMatrix4OneShared( const float4 *pInput, const int1 *pIndex, float4 *pMatrix, float4 *pOutput,  int sizeMax,  const float1 *pWeight, int sizeJoint )
{
	//size_t index = get_global_id(0) + get_global_id(1) *get_global_size(0);
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

	if( indexBase >= sizeMax )
		return;

#if ENABLE_MEMORY_SHARED
	__shared__		float4		shared_pMatrix_f4[ MATRIX_FIX_LENGTH * MATRIX_SIZE_LINE ];// 100个矩阵空间，6.4k
	if( threadIdx.x < sizeJoint )
	{
		for(int i=0;i<MATRIX_SIZE_LINE;i++)
			shared_pMatrix_f4[ threadIdx.x + i*sizeJoint ] = pMatrix[ threadIdx.x + i*sizeJoint ];	
	}
	__syncthreads();
#endif

	int index=indexBase;
#if SIZE_BLOCK_STATIC
		for( ; index<sizeMax; index+=blockDim.x * gridDim.x )
#endif
		{

			float4 pIn = pInput[index];
			float4 px = make_float4(pIn.x, pIn.x, pIn.x, pIn.x) ;
			float4 py = make_float4(pIn.y, pIn.y, pIn.y, pIn.y) ;
			float4 pz = make_float4(pIn.z, pIn.z, pIn.z, pIn.z) ;

			int offset = pIndex[index].x*4 ;

			float4 m0 = shared_pMatrix_f4[offset+0] ;
			float4 m1 = shared_pMatrix_f4[offset+1] ;
			float4 m2 = shared_pMatrix_f4[offset+2] ;
			float4 m3 = shared_pMatrix_f4[offset+3] ;

			pOutput[index] = px * m0 + py * m1 + pz * m2 + m3 ;
		}
}

__global__ void
transformVectorByMatrix4( const  float4 *pInput, const int *pIndex, float4 *pMatrix, float4 *pOutput,  int sizeMax,  const float *pWeight)
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

	if( indexBase >= sizeMax )
		return;

	int index=indexBase;
#if SIZE_BLOCK_STATIC
		for( ; index<sizeMax; index+=blockDim.x * gridDim.x )
#endif
		{
#if ENABLE_MEMORY_COALESCED
			int offset = pIndex[index]*4 ;
			float weight = pWeight[index] ;
#else
			int offset = pIndex[index*SIZE_PER_BONE+0]*4 ;
			float weight = pWeight[index*SIZE_PER_BONE+0] ;
#endif//#if ENABLE_MEMORY_COALESCED 合并访问

			float4 weight4 = make_float4( weight,weight,weight,weight ) ;

			float4 m0 = pMatrix[offset+0] * weight4 ;
			float4 m1 = pMatrix[offset+1] * weight4 ;
			float4 m2 = pMatrix[offset+2] * weight4 ;
			float4 m3 = pMatrix[offset+3] * weight4 ;

			
			for(int i=1;i<SIZE_PER_BONE; i++)
			{
#if ENABLE_MEMORY_COALESCED
				offset = pIndex[index+i*sizeMax]*4 ;
				weight = pWeight[index+i*sizeMax] ;
#else
				offset = pIndex[index*SIZE_PER_BONE+i]*4 ;
				weight = pWeight[index*SIZE_PER_BONE+i] ;
#endif//#if ENABLE_MEMORY_COALESCED 合并访问

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
}

__global__ void
transformVectorByMatrix4( const  Vector4 *pInput, const Vector1i *pIndex, Vector4 *pMatrix, Vector4 *pOutput,  int sizeMax,  const Vector1 *pWeight)
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

	if( indexBase >= sizeMax )
		return;

	int index=indexBase;
#if SIZE_BLOCK_STATIC
		for( ; index<sizeMax; index+=blockDim.x * gridDim.x )
#endif
		{
			int offset = pIndex[index*SIZE_PER_BONE+0].x*4 ;
			Vector1 weight = pWeight[index*SIZE_PER_BONE+0] ;
			Vector4 weight4 = make_vector4( weight.x,weight.x,weight.x,weight.x ) ;

			Vector4 m0 = pMatrix[offset+0] * weight4 ;
			Vector4 m1 = pMatrix[offset+1] * weight4 ;
			Vector4 m2 = pMatrix[offset+2] * weight4 ;
			Vector4 m3 = pMatrix[offset+3] * weight4 ;

			for(int i=1;i<SIZE_PER_BONE; i++)
			{
				offset = pIndex[index*SIZE_PER_BONE+i].x*4 ;
				weight = pWeight[index*SIZE_PER_BONE+i] ;
				weight4 = make_vector4( weight.x,weight.x,weight.x,weight.x ) ;

				m0 += pMatrix[offset+0] * weight4 ;
				m1 += pMatrix[offset+1] * weight4 ;
				m2 += pMatrix[offset+2] * weight4 ;
				m3 += pMatrix[offset+3] * weight4 ;
			}

			Vector4 pIn = pInput[index];
			Vector4 px = make_vector4(pIn.x, pIn.x, pIn.x, pIn.x) ;
			Vector4 py = make_vector4(pIn.y, pIn.y, pIn.y, pIn.y) ;
			Vector4 pz = make_vector4(pIn.z, pIn.z, pIn.z, pIn.z) ;

			pOutput[index] = px * m0 + py * m1 + pz * m2 + m3;
		}
}

__global__ void
transformVectorByMatrix4One( const float4 *pInput, const int1 *pIndex, float4 *pMatrix, float4 *pOutput,  int sizeMax,  const float1 *pWeight)
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

			int offset = pIndex[index].x*4 ;

			float4 m0 = pMatrix[offset+0] ;
			float4 m1 = pMatrix[offset+1] ;
			float4 m2 = pMatrix[offset+2] ;
			float4 m3 = pMatrix[offset+3] ;

			pOutput[index] = px * m0 + py * m1 + pz * m2 + m3 ;
		}
}

__global__ void
transformVectorByMatrix4OneSerial( const float4 *pInput, const int1 *pIndex, float4 *pMatrix, float4 *pOutput,  int sizeMax, int nElementPerThread, const float1 *pWeight)
{
	//size_t index = get_global_id(0) + get_global_id(1) *get_global_size(0);
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

	if( indexBase >= sizeMax )
		return;

	int index=indexBase*nElementPerThread;

		for( ; index<sizeMax && index<(indexBase+1)*nElementPerThread; index++ )
		{

			float4 pIn = pInput[index];
			float4 px = make_float4(pIn.x, pIn.x, pIn.x, pIn.x) ;
			float4 py = make_float4(pIn.y, pIn.y, pIn.y, pIn.y) ;
			float4 pz = make_float4(pIn.z, pIn.z, pIn.z, pIn.z) ;

			int offset = pIndex[index].x*4 ;

			float4 m0 = pMatrix[offset+0] ;
			float4 m1 = pMatrix[offset+1] ;
			float4 m2 = pMatrix[offset+2] ;
			float4 m3 = pMatrix[offset+3] ;

			pOutput[index] = px * m0 + py * m1 + pz * m2 + m3 ;
		}
}

__global__ void
transformVectorByMatrix4One( const Vector4 *pInput, const Vector1i *pIndex, Vector4 *pMatrix, Vector4 *pOutput,  int sizeMax,  const Vector1 *pWeight)
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

			Vector4 pIn = pInput[index];
			Vector4 px = make_vector4(pIn.x, pIn.x, pIn.x, pIn.x) ;
			Vector4 py = make_vector4(pIn.y, pIn.y, pIn.y, pIn.y) ;
			Vector4 pz = make_vector4(pIn.z, pIn.z, pIn.z, pIn.z) ;

			int offset = pIndex[index].x*4 ;

			Vector4 m0 = pMatrix[offset+0] ;
			Vector4 m1 = pMatrix[offset+1] ;
			Vector4 m2 = pMatrix[offset+2] ;
			Vector4 m3 = pMatrix[offset+3] ;

			pOutput[index] = px * m0 + py * m1 + pz * m2 + m3 ;
		}
}

extern "C" void
updateSharedMemory( const float* pHost )
{
#if ENABLE_MEMORY_CONST
	cudaMemcpyToSymbol( shared_pMatrix_f4, pHost, sizeof(float4) * MATRIX_SIZE_LINE * MATRIX_FIX_LENGTH );
#endif
}

extern "C" bool
runCUDADevice( const float *pInput, const int *pIndex, float *pMatrix, float *pOutput,  int sizeMax,  const float *pWeight, int sizeJoint )
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
#if ENABLE_MEMORY_CONST | ENABLE_MEMORY_SHARED

#if SIZE_PER_BONE==1
	 transformVectorByMatrix4OneShared<<< grid, block >>>( (FLOAT4*)pInput, (INT1*)pIndex, (FLOAT4*)pMatrix, (FLOAT4*)pOutput, sizeMax, (FLOAT1*)pWeight, sizeJoint );
#else
    transformVectorByMatrix4Shared<<< grid, block >>>( (FLOAT4*)pInput, (int*)pIndex, (FLOAT4*)pMatrix, (FLOAT4*)pOutput, sizeMax, (float*)pWeight, sizeJoint );
#endif// SIZE_PER_BONE==1
#else// !ENABLE_MEMORY_CONST
#if SIZE_PER_BONE==1
#if SERIAL_BLOCK_STATIC
	int nElementPerThread = (sizeMax + SIZE_BLOCK_X * SIZE_THREAD_X - 1) / (SIZE_BLOCK_X * SIZE_THREAD_X) ;

    transformVectorByMatrix4OneSerial<<< grid, block >>>( (FLOAT4*)pInput, (INT1*)pIndex, (FLOAT4*)pMatrix, (FLOAT4*)pOutput, sizeMax, nElementPerThread, (FLOAT1*)pWeight );
#else
    transformVectorByMatrix4One<<< grid, block >>>( (FLOAT4*)pInput, (INT1*)pIndex, (FLOAT4*)pMatrix, (FLOAT4*)pOutput, sizeMax, (FLOAT1*)pWeight );
#endif
#else
    transformVectorByMatrix4<<< grid, block >>>( (FLOAT4*)pInput, (int*)pIndex, (FLOAT4*)pMatrix, (FLOAT4*)pOutput, sizeMax, (float*)pWeight );
#endif
#endif

	return true;
}