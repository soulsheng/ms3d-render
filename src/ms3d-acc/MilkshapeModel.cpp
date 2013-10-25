/*
	MilkshapeModel.cpp

		Loads and renders a Milkshape3D model. 

	Author:	Brett Porter
	Email: brettporter@yahoo.com
	Website: http://www.geocities.com/brettporter/
	Copyright (C)2000, Brett Porter. All Rights Reserved.

	This file may be used only as long as this copyright notice remains intact.
*/

#include "stdafx.h"
#include "MilkshapeModel.h"
#include "common/utils.h"
#include <CL/cl_gl.h>

#include "SimpleOptimizations.cuh"

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

MilkshapeModel::MilkshapeModel()
{
	_idGPURenderItemsPerMesh = NULL;
}

MilkshapeModel::~MilkshapeModel()
{
#if CUDA_ENABLE
	// cuda 销毁
	cudaThreadExit();
#endif

	cudaGraphicsUnregisterResource(cuda_vbo_resource);

	//glBindBuffer(m_usNumMeshes+1, _idGPURenderItemsPerMesh);
	glDeleteBuffers(m_usNumMeshes+1, _idGPURenderItemsPerMesh);

	if (_idGPURenderItemsPerMesh)
	{
		delete[] _idGPURenderItemsPerMesh;
	}
}


bool MilkshapeModel::loadModelData( const char *filename )
{
	ifstream inputFile( filename, ios::in | ios::binary | ios::_Nocreate );
	if ( inputFile.fail())
		return false;	// "Couldn't open the model file."

	inputFile.seekg( 0, ios::end );
	long fileSize = inputFile.tellg();
	inputFile.seekg( 0, ios::beg );

	char *pBuffer = new char[fileSize];
	inputFile.read( pBuffer, fileSize );
	inputFile.close();

	const char *pPtr = pBuffer;
	MS3DHeader *pHeader = ( MS3DHeader* )pPtr;
	pPtr += sizeof( MS3DHeader );

	if ( strncmp( pHeader->m_ID, "MS3D000000", 10 ) != 0 )
	{
		delete [] pBuffer;
		return false; // "Not a valid Milkshape3D model file."
	}

	if ( pHeader->m_version < 3 || pHeader->m_version > 4 )
	{
		delete [] pBuffer;
		return false; // "Unhandled file version. Only Milkshape3D Version 1.3 and 1.4 is supported." );
	}

	//Read the vertices
	int nVertices = *( word* )pPtr; 
	m_usNumVerts = nVertices;
	m_pVertices = new Vertex[nVertices];
	pPtr += sizeof( word );

	int i;
	for ( i = 0; i < nVertices; i++ )
	{
		MS3DVertex *pVertex = ( MS3DVertex* )pPtr;
		m_pVertices[i].m_cBone = pVertex->m_cBone;
		memcpy( m_pVertices[i].m_vVert.Get(), pVertex->m_vertex, sizeof( float )*3 );
		pPtr += sizeof( MS3DVertex );
	}

	//Read the triangles
	int nTriangles = *( word* )pPtr;
	m_usNumTriangles = nTriangles;
	m_pTriangles = new Triangle[nTriangles];
	pPtr += sizeof( word );

	for ( i = 0; i < nTriangles; i++ )
	{
		MS3DTriangle *pTriangle = ( MS3DTriangle* )pPtr;
		int vertexIndices[3] = { pTriangle->m_usVertIndices[0], pTriangle->m_usVertIndices[1], pTriangle->m_usVertIndices[2] };
		float t[3] = { 1.0f-pTriangle->m_t.Get()[0], 1.0f-pTriangle->m_t.Get()[1], 1.0f-pTriangle->m_t.Get()[2] };
		memcpy( m_pTriangles[i].m_vNormals, pTriangle->m_vNormals, sizeof( float )*3*3 );
		memcpy( m_pTriangles[i].m_s, pTriangle->m_s.Get(), sizeof( float )*3 );
		memcpy( m_pTriangles[i].m_t, t, sizeof( float )*3 );
		memcpy( m_pTriangles[i].m_usVertIndices, vertexIndices, sizeof( int )*3 );
		pPtr += sizeof( MS3DTriangle );
	}

	//Load mesh groups
	int nGroups = *( word* )pPtr;
	m_usNumMeshes = nGroups;
	m_pMeshes = new Mesh[nGroups];
	pPtr += sizeof( word );

	unsigned short nTrianglesSub = 0;
	for ( i = 0; i < nGroups; i++ )
	{
		pPtr += sizeof( char );	// flags
		pPtr += 32;				// name

		nTrianglesSub = *( unsigned short* )pPtr;
		pPtr += sizeof( unsigned short );

		unsigned short usNumRepeatPadding = COUNT_MODEL_SIMULATE;
		m_pMeshes[i].m_usNumRepeatPadding = usNumRepeatPadding;
		int *pTriangleIndices = new int[nTrianglesSub*usNumRepeatPadding];
		for ( int j = 0; j < nTrianglesSub; j++ )
		{
			pTriangleIndices[j] = *( unsigned short* )pPtr;
			pPtr += sizeof( unsigned short );
		}

		// 填充 usNumRepeatPadding 倍面片
		for ( int j = 1; j < usNumRepeatPadding; j++ )
		{
			memcpy( pTriangleIndices + j * nTrianglesSub, pTriangleIndices, sizeof(int) * nTrianglesSub );
		}

		char materialIndex = *( char* )pPtr;
		pPtr += sizeof( char );
	
		m_pMeshes[i].m_materialIndex = materialIndex;
		m_pMeshes[i].m_usNumTris = nTrianglesSub*usNumRepeatPadding;
		m_pMeshes[i].m_uspIndices = pTriangleIndices;
	}

	//Read material information
	int nMaterials = *( word* )pPtr;
	m_numMaterials = nMaterials;
	m_pMaterials = new Material[nMaterials];
	pPtr += sizeof( word );
	for ( i = 0; i < nMaterials; i++ )
	{
		MS3DMaterial *pMaterial = ( MS3DMaterial* )pPtr;
		memcpy( m_pMaterials[i].m_ambient, pMaterial->m_ambient, sizeof( float )*4 );
		memcpy( m_pMaterials[i].m_diffuse, pMaterial->m_diffuse, sizeof( float )*4 );
		memcpy( m_pMaterials[i].m_specular, pMaterial->m_specular, sizeof( float )*4 );
		memcpy( m_pMaterials[i].m_emissive, pMaterial->m_emissive, sizeof( float )*4 );
		m_pMaterials[i].m_shininess = pMaterial->m_shininess;
		m_pMaterials[i].m_pTextureFilename = new char[strlen( pMaterial->m_texture )+1];
		strcpy( m_pMaterials[i].m_pTextureFilename, pMaterial->m_texture );
		pPtr += sizeof( MS3DMaterial );
	}

	reloadTextures();

	pPtr += 4;
	pPtr += 8;

	//Read in joint and animation info
	m_usNumJoints = *(unsigned short *)pPtr;
	pPtr += 2;
	//Allocate memory
	m_pJoints = new MS3DJoint[m_usNumJoints];
	//m_pJointsMatrix = new float[m_usNumJoints*16];
	m_pJointsMatrix = (float*) _aligned_malloc(m_usNumJoints*ELEMENT_COUNT_MATIRX * sizeof(float), 16);

	//Read in joint info
	for(int x = 0; x < m_usNumJoints; x++)
	{

		memcpy(&m_pJoints[x], pPtr, 93);//骨骼信息
		pPtr += 93;
		//Allocate memory 
		m_pJoints[x].m_RotKeyFrames = new MS3DKeyframe[m_pJoints[x].m_usNumRotFrames];

		m_pJoints[x].m_TransKeyFrames = new MS3DKeyframe[m_pJoints[x].m_usNumTransFrames];

		//copy keyframe information
		memcpy(m_pJoints[x].m_RotKeyFrames, pPtr, m_pJoints[x].m_usNumRotFrames * sizeof(MS3DKeyframe));
		pPtr += m_pJoints[x].m_usNumRotFrames * sizeof(MS3DKeyframe);
		memcpy(m_pJoints[x].m_TransKeyFrames, pPtr, m_pJoints[x].m_usNumTransFrames * sizeof(MS3DKeyframe));
		pPtr += m_pJoints[x].m_usNumTransFrames * sizeof(MS3DKeyframe);

	}


	//Find the parent joint array indices
	for(int x = 0; x < m_usNumJoints; x++)
	{
		//If the bone has a parent
		if(m_pJoints[x].m_cParent[0] != '\0')
		{
			//Compare names of theparent bone of x with the names of all bones
			for(int y = 0; y < m_usNumJoints; y++)
			{
				//A match has been found
				if(strcmp(m_pJoints[y].m_cName, m_pJoints[x].m_cParent) == 0)
				{
					m_pJoints[x].m_sParent = y;
				}
			}
		}
		//The bone has no parent
		else
		{
			m_pJoints[x].m_sParent = -1;
		}
	}


	//计算m_fTotalTime的数值，取平移与旋转关键帧的较大值
	float temp;
	for(int x = 0; x < m_usNumJoints; x++)
	{
		//Current joint
		MS3DJoint * pJoint = &m_pJoints[x];
		temp=pJoint->m_TransKeyFrames[pJoint->m_usNumTransFrames-1].m_fTime;
		if(m_fTotalTime<temp)m_fTotalTime=temp;
		temp=pJoint->m_RotKeyFrames[pJoint->m_usNumRotFrames-1].m_fTime;
		if(m_fTotalTime<temp)m_fTotalTime=temp;
	}


	delete[] pBuffer;

	Setup();


#if RENDERMODE_VBO
	initializeVBO();
#endif

	return true;
}

void MilkshapeModel::draw()
{
#if RENDERMODE_VBO
	renderVBO();
#else
	Model::draw();
#endif
}

void MilkshapeModel::initializeVBO()
{
	// 渲染点的vbo和CUDA id

#if ENABLE_MESH_MIX
	int nMeshCount = m_usNumMeshes+1;
#else
	int nMeshCount = m_usNumMeshes;
#endif

	_idGPURenderItemsPerMesh = new unsigned int[nMeshCount];

	glGenBuffersARB(nMeshCount, _idGPURenderItemsPerMesh);

	for (int i=0; i< nMeshCount; i++)
	{
		Ms3dVertexArrayMesh* pMesh = &m_meshVertexData.m_pMesh[i];

		int nSizeBufferVertex = ELEMENT_COUNT_POINT * pMesh->numOfVertex * sizeof(float);

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _idGPURenderItemsPerMesh[i]);

		glBufferDataARB(GL_ARRAY_BUFFER_ARB, nSizeBufferVertex, pMesh->pVertexArrayDynamic, GL_STATIC_DRAW_ARB);
	
	}

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, _idGPURenderItemsPerMesh[m_usNumMeshes], cudaGraphicsMapFlagsWriteDiscard));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

	// 面的vbo
	glGenBuffersARB(1, &_idVBOFaceIndexAll);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, _idVBOFaceIndexAll);

	// initialize buffer object
	int nSizeFaceIndex = m_meshVertexIndexTotal * sizeof(GLuint);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, nSizeFaceIndex, m_pIndexArray, GL_STATIC_DRAW_ARB);

}

void MilkshapeModel::renderVBO()
{

#if	RENDERMODE_MOVING

#if !ENABLE_TIMER_VBO_MAP
	getPlayTime( 1, 0, m_fTotalTime , true);

	updateJoints(fTime);
#endif

	modifyVBO();

#endif

	if(m_bDrawMesh)
	{
		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
		glEnable( GL_TEXTURE_2D );

		glDisable(GL_BLEND);
		glDisable(GL_ALPHA_TEST);

#if ENABLE_MESH_MIX
		int i = m_meshVertexData.m_numberOfMesh ;
#else
		for (int i=0; i< m_meshVertexData.m_numberOfMesh; i++)
#endif
		{
			glBindBufferARB( GL_ARRAY_BUFFER_ARB, _idGPURenderItemsPerMesh[i] );

			glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, _idVBOFaceIndexAll);

			glColor3f(1.0f, 1.0f, 1.0f);
			//glPointSize(2.0f);
#if  ENABLE_CROSSARRAY
			glVertexPointer( 3, GL_FLOAT, 32 , BUFFER_OFFSET( 20 ) );
			glTexCoordPointer( 2, GL_FLOAT, 32 , BUFFER_OFFSET( 0 ) );

			glEnableClientState( GL_TEXTURE_COORD_ARRAY );
			glEnableClientState( GL_VERTEX_ARRAY );
#else
			glVertexPointer( 3, GL_FLOAT, ELEMENT_COUNT_POINT*sizeof(float) , BUFFER_OFFSET( 0 ) );
			glEnableClientState( GL_VERTEX_ARRAY );
#endif

#if RENDERMODE_POINT
			glDrawArrays(GL_POINTS, 0, m_pMeshes[i].m_usNumTris * 3 );
#else
			glDrawElements( GL_TRIANGLES, m_meshVertexData.m_pMesh[i].numOfVertex, GL_UNSIGNED_INT, BUFFER_OFFSET(0));

#endif

#if  ENABLE_CROSSARRAY
			glDisableClientState( GL_TEXTURE_COORD_ARRAY );
#endif
			glDisableClientState( GL_VERTEX_ARRAY );

			glBindBufferARB( GL_ARRAY_BUFFER_ARB, NULL );
			glBindBufferARB( GL_ELEMENT_ARRAY_BUFFER_ARB, NULL );
		}
	}

}

void MilkshapeModel::modifyVBO()
{
#if ENABLE_NVIDIA_CUDA
	runCUDAHost();
#elif ENABLE_OPENCL_CPU
	ExecuteKernel( _context, _device_ID, _kernel, _cmd_queue );
#else
// 遍历每个Mesh，根据Joint更新每个Vertex的坐标
	int x=1;
#if !ENABLE_MESH_SINGLE
	#if ENABLE_MESH_MIX
		x=m_usNumMeshes;
	#else
		for ( x = 0; x < m_usNumMeshes; x++ )
	#endif
#endif
	{
		glBindBuffer( GL_ARRAY_BUFFER, _idGPURenderItemsPerMesh[x] );
		float* pVertexArrayDynamic = (float*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );

#if !ENABLE_TIMER_VBO_MAP
		
		float* pVertexArrayRaw = m_meshVertexData.m_pMesh[x].pVertexArrayRaw;

		int* pIndexJoint = m_meshVertexData.m_pMesh[x].pIndexJoint;
		float* pWeightJoint = m_meshVertexData.m_pMesh[x].pWeightJoint;

#if ENABLE_MESH_MIX
		int nSizeVertex = m_meshVertexIndexTotal;
#else
		int nSizeVertex = m_meshVertexData.m_pMesh[x].numOfVertex;
#endif

#if ENABLE_OPTIMIZE
	#if ENABLE_OPTIMIZE_SSE
			modifyVertexByJointKernelOptiSSE(  pVertexArrayRaw, pVertexArrayDynamic, pIndexJoint, pWeightJoint, nSizeVertex );
	#else
			modifyVertexByJointKernelOpti(  pVertexArrayRaw, pVertexArrayDynamic, pIndexJoint, pWeightJoint, nSizeVertex );
	#endif
#else
		modifyVertexByJointKernel(  pVertexArrayDynamic, pIndexJoint, pWeightJoint, pMesh );
#endif

#endif

		glUnmapBuffer( GL_ARRAY_BUFFER );
		glBindBuffer( GL_ARRAY_BUFFER, NULL );
	}

#endif

}

void MilkshapeModel::Setup()
{
	int x;
	static int cnt = 0;

	PreSetup();

#if 1
	//Go through each vertex
	for(x = 0; x < m_usNumVerts; x++)
	{
		//If there is no bone..
		if(m_pVertices[x].m_cBone== -1)
			continue;

		vgMs3d::CMatrix4X4 * mat = &m_pJoints[m_pVertices[x].m_cBone].m_matFinal;


		mat->InverseTranslateVec(m_pVertices[x].m_vVert.Get());
		mat->InverseRotateVec(m_pVertices[x].m_vVert.Get());
	}
#endif
	setupVertexArray();

	modifyVertexByJointInit();

}

void MilkshapeModel::PreSetup()
{
	//Go through each joint
	for(int x = 0; x < m_usNumJoints; x++)
	{

		m_pJoints[x].m_matLocal.SetRotation(m_pJoints[x].m_fRotation);
		m_pJoints[x].m_matLocal.SetTranslation(m_pJoints[x].m_fPosition);

		//Set the Abs transformations to the parents transformations, combined with their own local ones
		if(m_pJoints[x].m_sParent != -1)
		{
			m_pJoints[x].m_matAbs = m_pJoints[m_pJoints[x].m_sParent].m_matAbs * m_pJoints[x].m_matLocal;

		}
		//		//If there is no parent
		else
		{
			m_pJoints[x].m_matLocal.SetTranslation(m_pJoints[x].m_fPosition);
			m_pJoints[x].m_matAbs = m_pJoints[x].m_matLocal;

		}
		m_pJoints[x].m_matFinal = m_pJoints[x].m_matAbs;
	}
}


bool MilkshapeModel::ExecuteKernel( cl_context pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue )
{
#if !RENDERMODE_VBO
	return Model::ExecuteKernel(pContext, pDevice_ID, pKernel, pCmdQueue);
#endif

	cl_int err = CL_SUCCESS;
#if TIME_CL_MEMERY_WRITE
	// update matrix
	err = clEnqueueWriteBuffer(_cmd_queue, m_pfOCLMatrix, CL_TRUE, 0, sizeof(cl_float4) * MATRIX_SIZE_LINE * m_usNumJoints , m_pJointsMatrix, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("ERROR: Failed to clEnqueueReadBuffer...\n");
		return false;
	}
#endif

	//Set kernel arguments
	clSetKernelArg(_kernel, 2, sizeof(cl_mem), (void *) &m_pfOCLMatrix);

	int i = 1;
#if !ENABLE_MESH_SINGLE
#if ENABLE_MESH_MIX
	i=m_usNumMeshes;
#else
	for ( i = 0; i < m_usNumMeshes; i++ )
#endif
#endif
	{

#if ENABLE_MESH_MIX
		int nElementSize = m_meshVertexIndexTotal;
#else
		int nElementSize = m_pMeshes[i].m_usNumTris * 3;
#endif//ENABLE_MESH_MIX

#if  ENABLE_CL_GL_INTER
		clEnqueueAcquireGLObjects(_cmd_queue, 1, &m_oclKernelArg[i].m_pfOCLOutputBuffer, 0, 0, NULL);
#endif//ENABLE_CL_GL_INTER

#if TIME_CL_MEMERY_CALCULATE
		clSetKernelArg(_kernel, 0, sizeof(cl_mem), (void *) &m_oclKernelArg[i].m_pfInputBuffer);
		clSetKernelArg(_kernel, 1, sizeof(cl_mem), (void *) &m_oclKernelArg[i].m_pfOCLIndex);
		clSetKernelArg(_kernel, 3, sizeof(cl_mem), (void *) &m_oclKernelArg[i].m_pfOCLOutputBuffer);
		clSetKernelArg(_kernel, 4, sizeof(int), &nElementSize);
		clSetKernelArg(_kernel, 5, sizeof(cl_mem), (void *) &m_oclKernelArg[i].m_pfOCLWeight);

		cl_event g_perf_event = NULL;
		// execute kernel, pls notice g_bAutoGroupSize
#if LocalWorkSizeDef
		err= clEnqueueNDRangeKernel(_cmd_queue, _kernel, 2, NULL, m_oclKernelArg[i].globalWorkSize, NULL, 0, NULL, &g_perf_event);
#else
		err= clEnqueueNDRangeKernel(_cmd_queue, _kernel, 2, NULL, m_oclKernelArg[i].globalWorkSize, m_oclKernelArg[i].localWorkSize, 0, NULL, &g_perf_event);
#endif
		if (err != CL_SUCCESS)
		{
			printf("ERROR: Failed to execute kernel...\n");
			return false;
		}
		err = clWaitForEvents(1, &g_perf_event);
		if (err != CL_SUCCESS)
		{
			printf("ERROR: Failed to clWaitForEvents...\n");
			return false;
		}
#endif//TIME_CL_MEMERY_CALCULATE

#if  !ENABLE_CL_GL_INTER
		float* pVertexArrayDynamic = m_meshVertexData.m_pMesh[i].pVertexArrayDynamic;

		void* tmp_ptr = NULL;
		err = clEnqueueReadBuffer(_cmd_queue, m_oclKernelArg[i].m_pfOCLOutputBuffer, CL_TRUE, 0, sizeof(cl_float4) *	nElementSize , pVertexArrayDynamic, 0, NULL, NULL);

		if (err != CL_SUCCESS)
		{
			printf("ERROR: Failed to clEnqueueReadBuffer...\n");
			return false;
		}
#else
		clEnqueueReleaseGLObjects(_cmd_queue, 1, &m_oclKernelArg[i].m_pfOCLOutputBuffer, 0, 0, 0);
#endif//ENABLE_CL_GL_INTER
		clFinish(_cmd_queue);

	}

	return true;
}

void MilkshapeModel::SetupKernel( cl_context pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue )
{
#if !RENDERMODE_VBO
	Model::SetupKernel(pContext, pDevice_ID, pKernel, pCmdQueue);
	return;
#endif

	_context = pContext;
	_device_ID = pDevice_ID;
	_kernel = pKernel;
	_cmd_queue = pCmdQueue;

	const cl_mem_flags INFlags  = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY; 
	const cl_mem_flags OUTFlags = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE;

	m_pfOCLMatrix		= clCreateBuffer(_context, INFlags, sizeof(cl_float4) * MATRIX_SIZE_LINE	* m_usNumJoints ,	m_pJointsMatrix,	NULL); 

#if ENABLE_MESH_MIX
	int i=m_usNumMeshes;
#else
	for ( int i = 0; i < m_usNumMeshes; i++ )
#endif
	{
		float* pVertexArrayRaw = m_meshVertexData.m_pMesh[i].pVertexArrayRaw;
		float* pVertexArrayDynamic = m_meshVertexData.m_pMesh[i].pVertexArrayDynamic;

		int* pIndexJoint = m_meshVertexData.m_pMesh[i].pIndexJoint;
		float* pWeightJoint = m_meshVertexData.m_pMesh[i].pWeightJoint;

		OCLKernelArguments	&kernelArg = m_oclKernelArg[i];
		// allocate buffers
#if ENABLE_MESH_MIX
		int nElementSize = m_meshVertexIndexTotal;
#else
		int nElementSize = m_pMeshes[i].m_usNumTris * 3;
#endif
		kernelArg.m_pfInputBuffer		= clCreateBuffer(_context, INFlags,	sizeof(cl_float4) *	nElementSize,	pVertexArrayRaw,	NULL);
		//kernelArg.m_pfOCLOutputBuffer = clCreateBuffer(_context, OUTFlags,sizeof(cl_float4) *	nElementSize,	pVertexArrayDynamic,NULL);
		kernelArg.m_pfOCLIndex		= clCreateBuffer(_context, INFlags, sizeof(cl_int)	  *	nElementSize * SIZE_PER_BONE,	pIndexJoint,		NULL);   
		kernelArg.m_pfOCLWeight	= clCreateBuffer(_context, INFlags, sizeof(cl_int)	  *	nElementSize * SIZE_PER_BONE,	pWeightJoint,		NULL);   


		// thread size
		kernelArg.localWorkSize[0] = LocalWorkX;
		kernelArg.localWorkSize[1] = LocalWorkX;

		size_t  workGroupSizeMaximum;
		clGetKernelWorkGroupInfo(_kernel, _device_ID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *)&workGroupSizeMaximum, NULL);

		int nElementSizePadding = roundToPowerOf2( nElementSize );
		if ( nElementSizePadding > workGroupSizeMaximum )
		{
			kernelArg.globalWorkSize[0] = workGroupSizeMaximum;
			kernelArg.globalWorkSize[1] = nElementSizePadding / workGroupSizeMaximum;
		}

		cl_int errcode_ret;
		kernelArg.m_pfOCLOutputBuffer = clCreateFromGLBuffer(_context, CL_MEM_WRITE_ONLY, _idGPURenderItemsPerMesh[i], &errcode_ret);

		if ( CL_SUCCESS != errcode_ret )
		{
			//continue;
		}
	}

}

extern "C" bool
	runCUDADevice( const float *pInput, const int *pIndex, float *pMatrix, float *pOutput,  int sizeMax,  const float *pWeight );

bool MilkshapeModel::runCUDAHost()
{
	// map OpenGL buffer object for writing from CUDA
	float *d_pOutput;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_pOutput, &num_bytes,
		cuda_vbo_resource));

	// 骨骼矩阵
	int nBufferSize = ELEMENT_COUNT_MATIRX * m_usNumJoints * sizeof(float);
	cudaMemcpy( _cudaKernelArguments.d_pMatrix, m_pJointsMatrix, nBufferSize, cudaMemcpyHostToDevice );

	runCUDADevice(  _cudaKernelArguments.d_pInput, _cudaKernelArguments.d_pIndex, _cudaKernelArguments.d_pMatrix, 
		d_pOutput, m_meshVertexIndexTotal,  _cudaKernelArguments.d_pWeight );

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	return true;
}

void MilkshapeModel::initializeCUDA()
{
	int nElementSize = m_meshVertexIndexTotal;
	int nBufferSize = 0;

	Ms3dVertexArrayMesh* pMesh = &m_meshVertexData.m_pMesh[m_usNumMeshes];

	float	*h_pInput = pMesh->pVertexArrayRaw;
	float	*h_pOutput = pMesh->pVertexArrayDynamic;
	int		*h_pIndexJoint = pMesh->pIndexJoint;
	float	*h_pWeightJoint = pMesh->pWeightJoint;


	// 输入顶点坐标
	nBufferSize = nElementSize * sizeof(FLOAT4);
	cudaMalloc( &_cudaKernelArguments.d_pInput,  nBufferSize) ;
	cudaMemcpy( _cudaKernelArguments.d_pInput, h_pInput, nBufferSize, cudaMemcpyHostToDevice );

	// 输出变换后的顶点坐标
	//cudaMalloc( &_cudaKernelArguments.d_pOutput, nBufferSize ) ;
	//cudaMemcpy( _cudaKernelArguments.d_pOutput, h_pOutput, nBufferSize, cudaMemcpyHostToDevice );
 
	// 骨骼矩阵
	nBufferSize = MATRIX_SIZE_LINE * m_usNumJoints * sizeof(FLOAT4);
	cudaMalloc( &_cudaKernelArguments.d_pMatrix, nBufferSize ) ;
	cudaMemcpy( _cudaKernelArguments.d_pMatrix, m_pJointsMatrix, nBufferSize, cudaMemcpyHostToDevice );

	// 骨骼索引
	nBufferSize = SIZE_PER_BONE * nElementSize * sizeof(INT1);
	cudaMalloc( &_cudaKernelArguments.d_pIndex, nBufferSize ) ;
	cudaMemcpy( _cudaKernelArguments.d_pIndex, h_pIndexJoint, nBufferSize, cudaMemcpyHostToDevice );

	// 骨骼权重
	nBufferSize = SIZE_PER_BONE * nElementSize * sizeof(FLOAT1);
	cudaMalloc( &_cudaKernelArguments.d_pWeight, nBufferSize ) ;
	cudaMemcpy( _cudaKernelArguments.d_pWeight, h_pWeightJoint, nBufferSize, cudaMemcpyHostToDevice );

	

}


