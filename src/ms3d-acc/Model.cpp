/*
	Model.cpp

		Abstract base class for a model. The specific extended class will render the given model. 

	Author:	Brett Porter
	Email: brettporter@yahoo.com
	Website: http://www.geocities.com/brettporter/
	Copyright (C)2000, Brett Porter. All Rights Reserved.

	This file may be used only as long as this copyright notice remains intact.
*/
#include "stdafx.h"

#include "Model.h"
#include "ms3d-acc.h"
//using namespace vgMs3d;
#include <omp.h>			// OpenMP

#include <emmintrin.h>		// SSE

#define  LocalWorkX		8
#define  LocalWorkY		8


Model::Model()
{
	m_usNumMeshes = 0;
	m_pMeshes = NULL;
	m_numMaterials = 0;
	m_pMaterials = NULL;
	m_usNumTriangles = 0;
	m_pTriangles = NULL;
	m_usNumVerts = 0;
	m_pVertices = NULL;

	m_load = false;
	bFirstTime = true;
	m_bPlay = true;

#if ENABLE_DRAW
	m_bDrawBones = false;
	m_bDrawMesh = true;
#else
	m_bDrawBones = true;
	m_bDrawMesh = false;
#endif

	m_pIndexArray = NULL;
	maxMeshVertexNumber = 0;

	m_pJointsMatrix = NULL;
}

Model::~Model()
{
	int i;
	for ( i = 0; i < m_usNumMeshes; i++ )
		delete[] m_pMeshes[i].m_uspIndices;
	for ( i = 0; i < m_numMaterials; i++ )
		delete[] m_pMaterials[i].m_pTextureFilename;

	m_usNumMeshes = 0;
	if ( m_pMeshes != NULL )
	{
		delete[] m_pMeshes;
		m_pMeshes = NULL;
	}

	m_numMaterials = 0;
	if ( m_pMaterials != NULL )
	{
		delete[] m_pMaterials;
		m_pMaterials = NULL;
	}

	m_usNumTriangles = 0;
	if ( m_pTriangles != NULL )
	{
		delete[] m_pTriangles;
		m_pTriangles = NULL;
	}

	m_usNumVerts = 0;
	if ( m_pVertices != NULL )
	{
		delete[] m_pVertices;
		m_pVertices = NULL;
	}

	if (m_pIndexArray)
	{
		delete[] m_pIndexArray;
		m_pIndexArray = NULL;
	}

	if (m_pJointsMatrix)
	{
		_aligned_free(m_pJointsMatrix);//delete[] m_pJointsMatrix;
		m_pJointsMatrix = NULL;
	}
	
}

void Model::draw() 
{
#if	RENDERMODE_MOVING
	getPlayTime( 1, 0, m_fTotalTime , true);

	updateJoints(fTime);

	modifyVertexByJoint();
#endif

	GLboolean texEnabled = glIsEnabled( GL_TEXTURE_2D );
	
	if(m_bDrawMesh)
	{

		// Draw by group
		for ( int i = 0; i < m_usNumMeshes; i++ )
		{
#if ENABLE_CROSSARRAY
			glInterleavedArrays(GL_T2F_N3F_V3F, 0, m_meshVertexData.m_pMesh[i].pVertexArrayDynamic);
#else
			glVertexPointer(3, GL_FLOAT, ELEMENT_COUNT_POINT*sizeof(float), m_meshVertexData.m_pMesh[i].pVertexArrayDynamic);
			glEnableClientState( GL_VERTEX_ARRAY );
#endif


#if RENDERMODE_POINT
			glDrawArrays(GL_POINTS, 0, m_pMeshes[i].m_usNumTris * 3 );			
#else
			glDrawElements(GL_TRIANGLES, m_meshVertexData.m_pMesh[i].numOfVertex,
				GL_UNSIGNED_INT , m_pIndexArray);
#endif
			
#if !ENABLE_CROSSARRAY
			glDisableClientState( GL_VERTEX_ARRAY );
#endif

		}
	}



	if ( texEnabled )
		glEnable( GL_TEXTURE_2D );
	else
		glDisable( GL_TEXTURE_2D );
}

void Model::reloadTextures()
{
	for ( int i = 0; i < m_numMaterials; i++ )
		if ( strlen( m_pMaterials[i].m_pTextureFilename ) > 0 )
			m_pMaterials[i].m_texture = LoadGLTexture( m_pMaterials[i].m_pTextureFilename );
		else
			m_pMaterials[i].m_texture = 0;
}

void Model::updateJoints(float fTime)
{


	//std::cout << "Current Time: " << fTime << std::endl;
	// update matrix
	for(int i = 0; i < COUNT_MODEL_SIMULATE; i++)
	{
	// update matrix
	for(int x = 0; x < m_usNumJoints; x++)
	{
		//Transformation matrix
		vgMs3d::CMatrix4X4 matTmp;
		//Current joint
		MS3DJoint * pJoint = &m_pJoints[x];
		//Current frame]
		unsigned int uiFrame = 0;

		//if there are no keyframes, don't do any transformations
		if(pJoint->m_usNumRotFrames == 0 && pJoint->m_TransKeyFrames == 0)
		{
			pJoint->m_matFinal = pJoint->m_matAbs;
			continue;
		}
		//Calculate the current frame
		//Translation
		while(uiFrame < pJoint->m_usNumTransFrames && pJoint->m_TransKeyFrames[uiFrame].m_fTime < fTime)
			uiFrame++;

		float fTranslation[3];
		float fDeltaT = 1;
		float fInterp = 0;

		//If its at the extremes
		if(uiFrame == 0)
			memcpy(fTranslation, pJoint->m_TransKeyFrames[0].m_fParam, sizeof(float[3]));
		else if(uiFrame == pJoint->m_usNumTransFrames)
			memcpy(fTranslation, pJoint->m_TransKeyFrames[uiFrame-1].m_fParam, sizeof(float[3]));
		//If its in the middle of two frames
		else
		{
			MS3DKeyframe * pkCur = &pJoint->m_TransKeyFrames[uiFrame];
			MS3DKeyframe * pkPrev = &pJoint->m_TransKeyFrames[uiFrame-1];

			fDeltaT = pkCur->m_fTime - pkPrev->m_fTime;
			fInterp = (fTime - pkPrev->m_fTime) / fDeltaT;

			//Interpolate between the translations
			fTranslation[0] = pkPrev->m_fParam[0] + (pkCur->m_fParam[0] - pkPrev->m_fParam[0]) * fInterp;
			fTranslation[1] = pkPrev->m_fParam[1] + (pkCur->m_fParam[1] - pkPrev->m_fParam[1]) * fInterp;
			fTranslation[2] = pkPrev->m_fParam[2] + (pkCur->m_fParam[2] - pkPrev->m_fParam[2]) * fInterp;
		}
		//Calculate the current rotation
		uiFrame = 0;
		while(uiFrame < pJoint->m_usNumRotFrames && pJoint->m_RotKeyFrames[uiFrame].m_fTime < fTime)
			uiFrame++;


		//If its at the extremes
		if(uiFrame == 0)
			matTmp.SetRotation(pJoint->m_RotKeyFrames[0].m_fParam);
		else if(uiFrame == pJoint->m_usNumTransFrames)
			matTmp.SetRotation(pJoint->m_RotKeyFrames[uiFrame-1].m_fParam);
		//If its in the middle of two frames, use a quaternion SLERP operation to calculate a new position
		else
		{
			MS3DKeyframe * pkCur = &pJoint->m_RotKeyFrames[uiFrame];
			MS3DKeyframe * pkPrev = &pJoint->m_RotKeyFrames[uiFrame-1];

			fDeltaT = pkCur->m_fTime - pkPrev->m_fTime;
			fInterp = (fTime - pkPrev->m_fTime) / fDeltaT;

			//Create a rotation quaternion for each frame
			vgMs3d::CQuaternion qCur;
			vgMs3d::CQuaternion qPrev;
			qCur.FromEulers(pkCur->m_fParam);
			qPrev.FromEulers(pkPrev->m_fParam);
			//SLERP between the two frames
			vgMs3d::CQuaternion qFinal = SLERP(qPrev, qCur, fInterp);

			//Convert the quaternion to a rota tion matrix
			matTmp = qFinal.ToMatrix4();
		}

		//Set the translation part of the matrix
		matTmp.SetTranslation(fTranslation);

		//Calculate the joints final transformation
		vgMs3d::CMatrix4X4 matFinal = pJoint->m_matLocal * matTmp;

		//if there is no parent, just use the matrix you just made
		if(pJoint->m_sParent == -1)
			pJoint->m_matFinal = matFinal;
		//otherwise the final matrix is the parents final matrix * the new matrix
		else
			pJoint->m_matFinal = m_pJoints[pJoint->m_sParent].m_matFinal * matFinal;
	}//x

	for (int i=0;i<m_usNumJoints;i++)
	{

#if (ENABLE_OPTIMIZE_SSE)&&(ELEMENT_COUNT_POINT==3)

		for (int j=0;j<4;j++)
		{
			for (int k=0;k<4;k++)
			{
				float* pDst = m_pJointsMatrix+16*i;
				float* pSrc = m_pJoints[i].m_matFinal.Get();
				pDst[j*4+k] = pSrc[k*4+j];
			}
		}
#else
		memcpy(m_pJointsMatrix+16*i, m_pJoints[i].m_matFinal.Get(), sizeof(float)*16 );
#endif

	}

	}//i

}

void Model::getPlayTime(float fSpeed, float fStartTime, float fEndTime, bool bLoop)
{
	if (m_load==FALSE)
	{
//		Setup();
		m_load=TRUE;
		bFirstTime = true;
	}

	//First time animate has been called

	if(bFirstTime)
	{//修改的部分
		fLastTime= fStartTime;
		m_Timer.Init();
		m_Timer.GetSeconds();
		bFirstTime = false;
	}

	if (m_bPlay)
	{
		fTime = m_Timer.GetSeconds() * fSpeed;//帧时间
		fTime += fLastTime;
		fLastTime = fTime;
	}

	if(fTime > fEndTime)
	{
		if(bLoop)
		{			
			float dt = fEndTime - fStartTime;

			while (fTime > dt)
			{
				fTime -= dt;
			}

			fTime += fStartTime;

		}
		else
			fTime = fEndTime;
	}
}

void Model::modifyVertexByJointInit()
{
	
	// 遍历每个Mesh，根据Joint更新每个Vertex的坐标
	for(int x = 0; x < m_usNumMeshes; x++)
	{
		float* pVertexArrayStatic = m_meshVertexData.m_pMesh[x].pVertexArrayStatic;
		float* pVertexArrayDynamic = m_meshVertexData.m_pMesh[x].pVertexArrayDynamic;
		float* pVertexArrayRaw = m_meshVertexData.m_pMesh[x].pVertexArrayRaw;

		int* pIndexJoint = m_meshVertexData.m_pMesh[x].pIndexJoint;
		float* pWeightJoint = m_meshVertexData.m_pMesh[x].pWeightJoint;
		
		Mesh* pMesh = m_pMeshes+x;
		
		modifyVertexByJointInitKernel(  pVertexArrayRaw, pVertexArrayStatic, pIndexJoint, pWeightJoint, pMesh ); // 更新pVertexArrayStatic

#if !RENDERMODE_MOVING

		memcpy( pVertexArrayDynamic, pVertexArrayStatic,  ELEMENT_COUNT_POINT * m_pMeshes[x].m_usNumTris * 3 * sizeof(float) );

#endif
	}
}

void Model::modifyVertexByJoint()
{
	
	// 遍历每个Mesh，根据Joint更新每个Vertex的坐标
	//int x = 2;
	for(int x = 0; x < m_usNumMeshes; x++)
	{
		float* pVertexArrayRaw = m_meshVertexData.m_pMesh[x].pVertexArrayRaw;
		float* pVertexArrayDynamic = m_meshVertexData.m_pMesh[x].pVertexArrayDynamic;

		int* pIndexJoint = m_meshVertexData.m_pMesh[x].pIndexJoint;
		float* pWeightJoint = m_meshVertexData.m_pMesh[x].pWeightJoint;
		
		Mesh* pMesh = m_pMeshes+x;
		
#if ENABLE_OPTIMIZE

#if ENABLE_OPTIMIZE_SSE
		modifyVertexByJointKernelOptiSSE(  pVertexArrayRaw, pVertexArrayDynamic, pIndexJoint, pWeightJoint, pMesh );
#else
		modifyVertexByJointKernelOpti(  pVertexArrayRaw, pVertexArrayDynamic, pIndexJoint, pWeightJoint, pMesh );
#endif

#else
		modifyVertexByJointKernel(  pVertexArrayDynamic, pIndexJoint, pMesh );
#endif
	}
}

void Model::setupVertexArray()
{
	// 根据Mesh数生成m_usNumMeshes个VertexArray.
	// 在m_meshVertexData的析构函数中释放.
	m_meshVertexData.m_pMesh = new Ms3dVertexArrayMesh[m_usNumMeshes];
	m_meshVertexData.m_numberOfMesh = m_usNumMeshes;


	for(int x = 0; x < m_usNumMeshes; x++)
	{
		// 在m_pMesh的析构函数中释放. 如需要增加法线则使用
		int* pIndexJoint = new int[m_pMeshes[x].m_usNumTris * 3 * SIZE_PER_BONE];
		float* pWeightJoint = new float[m_pMeshes[x].m_usNumTris * 3 * SIZE_PER_BONE];

		int nVertexSize = ELEMENT_COUNT_POINT * m_pMeshes[x].m_usNumTris * 3;

		float* pVertexArrayStatic = (float*)_aligned_malloc( nVertexSize*sizeof(float), 16 );//new float[];
		float* pVertexArrayDynamic = (float*)_aligned_malloc( nVertexSize*sizeof(float), 16 );//new float[];
		float* pVertexArrayRaw = (float*)_aligned_malloc( nVertexSize*sizeof(float), 16 );//new float[];

		m_meshVertexData.m_pMesh[x].numOfVertex = m_pMeshes[x].m_usNumTris * 3;
		m_meshVertexData.m_pMesh[x].pVertexArrayStatic = pVertexArrayStatic;
		m_meshVertexData.m_pMesh[x].pVertexArrayDynamic = pVertexArrayDynamic;
		m_meshVertexData.m_pMesh[x].pIndexJoint = pIndexJoint;
		m_meshVertexData.m_pMesh[x].pWeightJoint = pWeightJoint;
		m_meshVertexData.m_pMesh[x].pVertexArrayRaw = pVertexArrayRaw;
		m_meshVertexData.m_pMesh[x].materialID = m_pMeshes[x].m_materialIndex;

		if (m_meshVertexData.m_pMesh[x].numOfVertex > (int)maxMeshVertexNumber)
		{
			maxMeshVertexNumber = m_meshVertexData.m_pMesh[x].numOfVertex;
		}
	}
	
	/*vgMs3d::CVector3 vecNormal;
	vgMs3d::CVector3 vecVertex;


	for(int x = 0; x < m_usNumMeshes; x++)
	{
		int vertexCnt = 0;

		float* pVertexArrayStatic = m_meshVertexData.m_pMesh[x].pVertexArrayStatic;
		int* pIndexJoint = m_meshVertexData.m_pMesh[x].pIndexJoint;

		for(int y = 0; y < m_pMeshes[x].m_usNumTris; y++)
		{
			//Set triangle pointer to triangle #1

			Triangle * pTri = &m_pTriangles[m_pMeshes[x].m_uspIndices[y]];
			//Loop through each vertex 
			for(int z = 0; z < 3; z++)
			{
				//Get the vertex
				Vertex * pVert = &m_pVertices[pTri->m_usVertIndices[z]];
				pIndexJoint[3*y+z] = pVert->m_cBone;

				pVertexArrayStatic[vertexCnt++] = pTri->m_s[z];
				pVertexArrayStatic[vertexCnt++] = 1.0f - pTri->m_t[z];

				// 不初始化法线
				pVertexArrayStatic[vertexCnt++] = pTri->m_vNormals[z].Get()[0];
				pVertexArrayStatic[vertexCnt++] = pTri->m_vNormals[z].Get()[1];
				pVertexArrayStatic[vertexCnt++] = pTri->m_vNormals[z].Get()[2];

				pVertexArrayStatic[vertexCnt++] = pVert->m_vVert.Get()[0];
				pVertexArrayStatic[vertexCnt++] = pVert->m_vVert.Get()[1];
				pVertexArrayStatic[vertexCnt++] = pVert->m_vVert.Get()[2];
			}
		}
	}*/		

	m_pIndexArray = new unsigned int[maxMeshVertexNumber];

	for (unsigned int i=0; i<maxMeshVertexNumber; i++)
	{
		m_pIndexArray[i] = i;
	}
}

void Model::modifyVertexByJointKernel( float* pVertexArrayDynamic  , int* pIndexJoint, Mesh* pMesh)
{
	vgMs3d::CVector3 vecNormal;
	vgMs3d::CVector3 vecVertex;

	int vertexCnt = 0;

	//遍历Mesh的每个三角面
	for(int y = 0; y < pMesh->m_usNumTris; y++)
	{
		//Set triangle pointer to triangle #1

		Triangle * pTri = &m_pTriangles[pMesh->m_uspIndices[y]];
		// 遍历三角面的三个顶点 
		for(int z = 0; z < 3; z++)
		{
			//Get the vertex
			Vertex * pVert = &m_pVertices[pTri->m_usVertIndices[z]];

			//If it has no bone, render as is
			if(pVert->m_cBone == -1)
			{
				//Send all 3 components without modification
				vecNormal = pTri->m_vNormals[z];
				vecVertex = pVert->m_vVert;
			}
			//Otherwise, transform the vertices and normals before displaying them
			else
			{
				MS3DJoint * pJoint = &m_pJoints[pVert->m_cBone];
				// Transform the normals
				// vecNormal = pTri->m_vNormals[z];
				// Only rotate it, no translation
				// 当前版本不计算法线					
				// vecNormal.Transform3(pJoint->m_matFinal);

				// Transform the vertex
				vecVertex = pVert->m_vVert;
				// translate as well as rotate
				vecVertex.Transform4(pJoint->m_matFinal);

			}
#if ENABLE_CROSSARRAY
			vertexCnt += 2;

			// 法线没有被计算和拷贝
			pVertexArrayDynamic[vertexCnt++] = vecNormal[0];
			pVertexArrayDynamic[vertexCnt++] = vecNormal[1];
			pVertexArrayDynamic[vertexCnt++] = vecNormal[2];
#endif
			pVertexArrayDynamic[vertexCnt++] = vecVertex[0];
			pVertexArrayDynamic[vertexCnt++] = vecVertex[1];
			pVertexArrayDynamic[vertexCnt++] = vecVertex[2];
		}//for z
	}//for y
}

void Model::modifyVertexByJointInitKernel( float* pVertexArrayStatic , float* pVertexArrayDynamic  , int* pIndexJoint, float* pWeightJoint,Mesh* pMesh)
{
	vgMs3d::CVector3 vecNormal;
	vgMs3d::CVector3 vecVertex;

	int vertexCnt = 0;

	//遍历Mesh的每个三角面
	for(int y = 0; y < pMesh->m_usNumTris; y++)
	{
		//Set triangle pointer to triangle #1

		Triangle * pTri = &m_pTriangles[pMesh->m_uspIndices[y]];
		// 遍历三角面的三个顶点 
		for(int z = 0; z < 3; z++)
		{
			//Get the vertex
			Vertex * pVert = &m_pVertices[pTri->m_usVertIndices[z]];
			
			for (int i=0;i<SIZE_PER_BONE;i++)
			{
				pIndexJoint[(3*y+z)*SIZE_PER_BONE + i] = pVert->m_cBone;
				pWeightJoint[(3*y+z)*SIZE_PER_BONE + i] = 1.0f/SIZE_PER_BONE;
			}

			//If it has no bone, render as is
			if(pVert->m_cBone == -1)
			{
				//Send all 3 components without modification
				vecNormal = pTri->m_vNormals[z];
				vecVertex = pVert->m_vVert;
			}
			//Otherwise, transform the vertices and normals before displaying them
			else
			{
				MS3DJoint * pJoint = &m_pJoints[pVert->m_cBone];
				
				vecVertex = pVert->m_vVert;
				// translate as well as rotate
				vecVertex.Transform4(pJoint->m_matFinal);

			}
#if ENABLE_CROSSARRAY
			vertexCnt += 2;

			// 法线没有被计算和拷贝
			pVertexArrayDynamic[vertexCnt++] = vecNormal[0];
			pVertexArrayDynamic[vertexCnt++] = vecNormal[1];
			pVertexArrayDynamic[vertexCnt++] = vecNormal[2];
#endif
			pVertexArrayDynamic[vertexCnt++] = vecVertex[0];
			pVertexArrayDynamic[vertexCnt++] = vecVertex[1];
			pVertexArrayDynamic[vertexCnt++] = vecVertex[2];

#if ENABLE_CROSSARRAY
			if(pVertexArrayStatic)
			{
				pVertexArrayStatic[ ELEMENT_COUNT_POINT*(3*y+z) +5 ] = pVert->m_vVert[0];
				pVertexArrayStatic[ ELEMENT_COUNT_POINT*(3*y+z) +6 ] = pVert->m_vVert[1];
				pVertexArrayStatic[ ELEMENT_COUNT_POINT*(3*y+z) +7 ] = pVert->m_vVert[2];
			}
#else
			if(pVertexArrayStatic)
			{
				pVertexArrayStatic[ ELEMENT_COUNT_POINT*(3*y+z) ] = pVert->m_vVert[0];
				pVertexArrayStatic[ ELEMENT_COUNT_POINT*(3*y+z)+1 ] = pVert->m_vVert[1];
				pVertexArrayStatic[ ELEMENT_COUNT_POINT*(3*y+z)+2 ] = pVert->m_vVert[2];
			}
#endif
		}//for z
	}//for y
}


void Model::modifyVertexByJointKernelOpti( float* pVertexArrayRaw , float* pVertexArrayDynamic , int* pIndexJoint, float* pWeightJoint, Mesh* pMesh)
{


	float *pSrcPos = pVertexArrayRaw +STRIDE_POINT;
	float *pDestPos = pVertexArrayDynamic + STRIDE_POINT;

	//遍历每个顶点
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
	for(int y = 0; y < pMesh->m_usNumTris*3; y++)
	{
#if 0
		float* pSrcPosOne = pSrcPos+ELEMENT_COUNT_POINT*y;
		float* pDestPosOne = pDestPos+ELEMENT_COUNT_POINT*y;
		float* pMatOne = m_pJointsMatrix+ELEMENT_COUNT_MATIRX*pIndexJoint[y];

		kernelElement( pSrcPosOne, pDestPosOne, pMatOne );
#else
		float* pIn = pSrcPos+ELEMENT_COUNT_POINT*y;
		float* pOut = pDestPos+ELEMENT_COUNT_POINT*y;
		
#if (SIZE_PER_BONE==1)
		float* pMat = m_pJointsMatrix+ELEMENT_COUNT_MATIRX*pIndexJoint[y];
#else
		float  pMat[16];
		memset(pMat, 0, 16*sizeof(float) );
		for (int i=0;i<SIZE_PER_BONE;i++)
		{
			float* pMatOne = m_pJointsMatrix+ELEMENT_COUNT_MATIRX*pIndexJoint[y*SIZE_PER_BONE+i];
			float weight = pWeightJoint[y*SIZE_PER_BONE+i];
			for (int j=0;j<16;j++)
			{
				pMat[j] += pMatOne[j] * weight;
			}
		}
#endif

		pOut[0] =
			(pMat[0*4+0] * pIn[0] +
			pMat[1*4+0] * pIn[1] +
			pMat[2*4+0] * pIn[2] +
			pMat[3*4+0]) ;

		pOut[1] =
			(pMat[0*4+1] * pIn[0] +
			pMat[1*4+1] * pIn[1] +
			pMat[2*4+1] * pIn[2] +
			pMat[3*4+1]) ;

		pOut[2] =
			(pMat[0*4+2] * pIn[0] +
			pMat[1*4+2] * pIn[1] +
			pMat[2*4+2] * pIn[2] +
			pMat[3*4+2]) ;
#endif
	}//for y
}

void Model::kernelElement( float* pIn, float* pOut, float* pMat )
{
	pOut[0] =
		(pMat[0*4+0] * pIn[0] +
		pMat[1*4+0] * pIn[1] +
		pMat[2*4+0] * pIn[2] +
		pMat[3*4+0]) ;

	pOut[1] =
		(pMat[0*4+1] * pIn[0] +
		pMat[1*4+1] * pIn[1] +
		pMat[2*4+1] * pIn[2] +
		pMat[3*4+1]) ;

	pOut[2] =
		(pMat[0*4+2] * pIn[0] +
		pMat[1*4+2] * pIn[1] +
		pMat[2*4+2] * pIn[2] +
		pMat[3*4+2]) ;
}

// --SSE----------------------------------------------------------------------------------


/** Performing the transpose of a 4x4 matrix of single precision floating
    point values.
    Arguments r0, r1, r2, and r3 are __m128 values whose elements
    form the corresponding rows of a 4x4 matrix.
    The matrix transpose is returned in arguments r0, r1, r2, and
    r3 where r0 now holds column 0 of the original matrix, r1 now
    holds column 1 of the original matrix, etc.
*/
#define __MM_TRANSPOSE4x4_PS(r0, r1, r2, r3)                                            \
    {                                                                                   \
        __m128 tmp3, tmp2, tmp1, tmp0;                                                  \
                                                                                        \
                                                            /* r00 r01 r02 r03 */       \
                                                            /* r10 r11 r12 r13 */       \
                                                            /* r20 r21 r22 r23 */       \
                                                            /* r30 r31 r32 r33 */       \
                                                                                        \
        tmp0 = _mm_unpacklo_ps(r0, r1);                       /* r00 r10 r01 r11 */     \
        tmp2 = _mm_unpackhi_ps(r0, r1);                       /* r02 r12 r03 r13 */     \
        tmp1 = _mm_unpacklo_ps(r2, r3);                       /* r20 r30 r21 r31 */     \
        tmp3 = _mm_unpackhi_ps(r2, r3);                       /* r22 r32 r23 r33 */     \
                                                                                        \
        r0 = _mm_movelh_ps(tmp0, tmp1);                         /* r00 r10 r20 r30 */   \
        r1 = _mm_movehl_ps(tmp1, tmp0);                         /* r01 r11 r21 r31 */   \
        r2 = _mm_movelh_ps(tmp2, tmp3);                         /* r02 r12 r22 r32 */   \
        r3 = _mm_movehl_ps(tmp3, tmp2);                         /* r03 r13 r23 r33 */   \
    }

/// Accumulate four vector of single precision floating point values.
#define __MM_ACCUM4_PS(a, b, c, d)                                                  \
	_mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d))

/** Performing dot-product between four vector and three vector of single
    precision floating point values.
*/
#define __MM_DOT4x3_PS(r0, r1, r2, r3, v0, v1, v2)                                  \
    __MM_ACCUM4_PS(_mm_mul_ps(r0, v0), _mm_mul_ps(r1, v1), _mm_mul_ps(r2, v2), r3)

/// Same as _mm_load_ps, but can help VC generate more optimised code.
#define __MM_LOAD_PS(p)                                                             \
	(*(__m128*)(p))

/// Same as _mm_store_ps, but can help VC generate more optimised code.
#define __MM_STORE_PS(p, v)                                                         \
	(*(__m128*)(p) = (v))


/// Calculate multiply of two vector and plus another vector
#define __MM_MADD_PS(a, b, c)                                                       \
	_mm_add_ps(_mm_mul_ps(a, b), c)

/// Linear interpolation
#define __MM_LERP_PS(t, a, b)                                                       \
	__MM_MADD_PS(_mm_sub_ps(b, a), t, a)
//---------------------------------------------------------------------
// Some useful macro for collapse matrices.
//---------------------------------------------------------------------
#if STRUCT_OCL
#define __LOAD_MATRIX(row0, row1, row2, pMatrix)                        \
	{                                                                   \
	row0 = __MM_LOAD_PS(pMatrix[0]);                             \
	row1 = __MM_LOAD_PS(pMatrix[1]);                             \
	row2 = __MM_LOAD_PS(pMatrix[2]);                             \
	}

#define __LERP_MATRIX(row0, row1, row2, weight, pMatrix)                \
	{                                                                   \
	row0 = __MM_LERP_PS(weight, row0, __MM_LOAD_PS(pMatrix[0])); \
	row1 = __MM_LERP_PS(weight, row1, __MM_LOAD_PS(pMatrix[1])); \
	row2 = __MM_LERP_PS(weight, row2, __MM_LOAD_PS(pMatrix[2])); \
	}

#define __LOAD_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix)       \
	{                                                                   \
	row0 = _mm_mul_ps(__MM_LOAD_PS(pMatrix[0]), weight);         \
	row1 = _mm_mul_ps(__MM_LOAD_PS(pMatrix[1]), weight);         \
	row2 = _mm_mul_ps(__MM_LOAD_PS(pMatrix[2]), weight);         \
	}

#define __ACCUM_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix)      \
	{                                                                   \
	row0 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix[0]), weight, row0); \
	row1 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix[1]), weight, row1); \
	row2 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix[2]), weight, row2); \
	}
#else

#define __LOAD_MATRIX(row0, row1, row2, pMatrix)                        \
	{                                                                   \
	row0 = __MM_LOAD_PS(pMatrix);                             \
	row1 = __MM_LOAD_PS(pMatrix+4);                             \
	row2 = __MM_LOAD_PS(pMatrix+8);                             \
	}

#define __LERP_MATRIX(row0, row1, row2, weight, pMatrix)                \
	{                                                                   \
	row0 = __MM_LERP_PS(weight, row0, __MM_LOAD_PS(pMatrix)); \
	row1 = __MM_LERP_PS(weight, row1, __MM_LOAD_PS(pMatrix+4)); \
	row2 = __MM_LERP_PS(weight, row2, __MM_LOAD_PS(pMatrix+8)); \
	}

#define __LOAD_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix)       \
	{                                                                   \
	row0 = _mm_mul_ps(__MM_LOAD_PS(pMatrix), weight);         \
	row1 = _mm_mul_ps(__MM_LOAD_PS(pMatrix+4), weight);         \
	row2 = _mm_mul_ps(__MM_LOAD_PS(pMatrix+8), weight);         \
	}

#define __ACCUM_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix)      \
	{                                                                   \
	row0 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix), weight, row0); \
	row1 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix+4), weight, row1); \
	row2 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix+8), weight, row2); \
	}

#endif

/** Fill vector of single precision floating point with selected value.
    Argument 'fp' is a digit[0123] that represents the fp of argument 'v'.
*/
#define __MM_SELECT(v, fp)                                                          \
    _mm_shuffle_ps((v), (v), _MM_SHUFFLE((fp),(fp),(fp),(fp)))

/** Collapse one-weighted matrix.
    Eliminated multiply by weight since the weight should be equal to one always
*/
#define __COLLAPSE_MATRIX_W1(row0, row1, row2, ppMatrices, pIndices, pWeights)  \
    {                                                                           \
        pMatrix0 = blendMatrices +pIndices[0]*MATRIX_SIZE_LINE*4;                                  \
        __LOAD_MATRIX(row0, row1, row2, pMatrix0);                              \
    }


/// Calculate multiply of two vector and plus another vector
__m128 MM_MADD_PS(__m128 a, __m128 b, __m128 c)                                                      
{
	return _mm_add_ps(_mm_mul_ps(a, b), c);
}

/// Linear interpolation
__m128 MM_LERP_PS(__m128 t, __m128 a, __m128 b)                                                      
{
	return MM_MADD_PS(_mm_sub_ps(b, a), t, a);
}

void LOAD_MATRIX(__m128*row0, __m128*row1, __m128*row2, float *pMatrix)                        
	{                                                                   
	*row0 = *(__m128*)(pMatrix);                             
	*row1 = *(__m128*)(pMatrix+4);                             
	*row2 = *(__m128*)(pMatrix+8);                             
	}

void LERP_MATRIX(__m128*row0, __m128*row1, __m128*row2, __m128 *weight, float *pMatrix)                
	{                                                                   
	*row0 = MM_LERP_PS(*weight, *row0, *(__m128*)(pMatrix)); 
	*row1 = MM_LERP_PS(*weight, *row1, *(__m128*)(pMatrix+4)); 
	*row2 = MM_LERP_PS(*weight, *row2, *(__m128*)(pMatrix+8)); 
	}

void LOAD_WEIGHTED_MATRIX(__m128*row0, __m128*row1, __m128*row2, __m128 *weight, float *pMatrix)       
	{                                                                   
	*row0 = _mm_mul_ps(*(__m128*)(pMatrix), *weight);         
	*row1 = _mm_mul_ps(*(__m128*)(pMatrix+4), *weight);         
	*row2 = _mm_mul_ps(*(__m128*)(pMatrix+8), *weight);         
	}

void ACCUM_WEIGHTED_MATRIX(__m128*row0, __m128*row1, __m128*row2, __m128 *weight, float *pMatrix)      
	{                                                                   
	*row0 = MM_MADD_PS(*(__m128*)(pMatrix), *weight, *row0); 
	*row1 = MM_MADD_PS(*(__m128*)(pMatrix+4), *weight, *row1); 
	*row2 = MM_MADD_PS(*(__m128*)(pMatrix+8), *weight, *row2); 
	}

void  COLLAPSE_MATRIX_W1(__m128*row0, __m128*row1, __m128*row2, float *ppMatrices, unsigned short *pIndices, float *pWeights)  
	{                                                                           
		float * pMatrix0;               
	pMatrix0 = ppMatrices +pIndices[0]*MATRIX_SIZE_LINE*4;                                  
	LOAD_MATRIX(row0, row1, row2, pMatrix0);                             
	}

void  COLLAPSE_MATRIX_W2(__m128*row0, __m128*row1, __m128*row2, float *ppMatrices, unsigned short *pIndices, float *pWeights) 
    {                                                                          
		__m128 weight = _mm_load_ps1(pWeights + 1);                                   
		float * pMatrix0, *pMatrix1;               
        pMatrix0 = ppMatrices +pIndices[0]*MATRIX_SIZE_LINE*4;                                    
        LOAD_MATRIX(row0, row1, row2, pMatrix0);                              
        pMatrix1 = ppMatrices +pIndices[1]*MATRIX_SIZE_LINE*4;                                    
        LERP_MATRIX(row0, row1, row2, &weight, pMatrix1);                      
    }

/** Collapse three-weighted matrix.
*/
void  COLLAPSE_MATRIX_W3(__m128*row0, __m128*row1, __m128*row2, float *ppMatrices, unsigned short *pIndices, float *pWeights)  
    {                                                                           
        __m128 weight = _mm_load_ps1(pWeights + 0);                                    
		float * pMatrix0, *pMatrix1, *pMatrix2;               
        pMatrix0 = ppMatrices + pIndices[0]*MATRIX_SIZE_LINE*4;                                     
        LOAD_WEIGHTED_MATRIX(row0, row1, row2, &weight, pMatrix0);             
        weight = _mm_load_ps1(pWeights + 1);                                    
        pMatrix1 = ppMatrices + pIndices[1]*MATRIX_SIZE_LINE*4;                                    
        ACCUM_WEIGHTED_MATRIX(row0, row1, row2, &weight, pMatrix1);           
        weight = _mm_load_ps1(pWeights + 2);                                   
        pMatrix2 = ppMatrices + pIndices[2]*MATRIX_SIZE_LINE*4;                                     
        ACCUM_WEIGHTED_MATRIX(row0, row1, row2, &weight, pMatrix2);            
    }

/** Collapse four-weighted matrix.
*/
void COLLAPSE_MATRIX_W4(__m128*row0, __m128*row1, __m128*row2, float *ppMatrices, unsigned short *pIndices, float *pWeights)  
    {                                                                           
        /* Load four blend weights at one time, they will be shuffled later */  
        __m128 weights = _mm_loadu_ps(pWeights);                                       
                                                                                
		float * pMatrix0, *pMatrix1, *pMatrix2, *pMatrix3;               
        pMatrix0 = ppMatrices + pIndices[0]*MATRIX_SIZE_LINE*4;                                     
         __m128 weight = __MM_SELECT(weights, 0);                                      
        LOAD_WEIGHTED_MATRIX(row0, row1, row2, &weight, pMatrix0);             
        pMatrix1 = ppMatrices + pIndices[1]*MATRIX_SIZE_LINE*4;                                    
        weight = __MM_SELECT(weights, 1);                                       
        ACCUM_WEIGHTED_MATRIX(row0, row1, row2, &weight, pMatrix1);            
        pMatrix2 = ppMatrices + pIndices[2]*MATRIX_SIZE_LINE*4;                                    
        weight = __MM_SELECT(weights, 2);                                       
        ACCUM_WEIGHTED_MATRIX(row0, row1, row2, &weight, pMatrix2);           
        pMatrix3 = ppMatrices + pIndices[3]*MATRIX_SIZE_LINE*4;                                     
        weight = __MM_SELECT(weights, 3);                                      
        ACCUM_WEIGHTED_MATRIX(row0, row1, row2, &weight, pMatrix3);            
    }

void collapseOneMatrix( __m128* m00, __m128*m01, __m128*m02, float *blendMatrices)                                                   
{       
	* m00 = *(__m128*)(blendMatrices + 0);
	* m01 = *(__m128*)(blendMatrices + 4);
	* m02 = *(__m128*)(blendMatrices + 8);
}


void collapseOneMatrix( __m128* m00, __m128*m01, __m128*m02,                                                     
	float *pBlendWeight, unsigned short *pBlendIndex,                                             
	float *blendMatrices,                                                         
	int blendWeightStride, int blendIndexStride,                                    
	int numWeightsPerVertex)                                                   
	{                                                                           
	/* Important Note: If reuse pMatrixXXX frequently, M$ VC7.1 will */    
	/* generate wrong code here!!!                                   */    
				unsigned short matrixIndex1 = pBlendIndex[0 ]*MATRIX_SIZE_LINE*4;

				__m128 weight = _mm_load_ps1(pBlendWeight);                                   

				* m00 = _mm_mul_ps( *(__m128*)(blendMatrices+matrixIndex1+0) , weight ); 
				* m01 = _mm_mul_ps( *(__m128*)(blendMatrices+matrixIndex1+4) , weight ); 
				* m02 = _mm_mul_ps( *(__m128*)(blendMatrices+matrixIndex1+8) , weight ); 


				for(int i=1; i<SIZE_PER_BONE; i++)
				{
					matrixIndex1 = pBlendIndex[ i ]*MATRIX_SIZE_LINE*4;
					weight = _mm_load_ps1(pBlendWeight+i);  

					* m00 = _mm_add_ps( _mm_mul_ps( *(__m128*)(blendMatrices+matrixIndex1+0) ,  weight ),  * m00 );
					* m01 = _mm_add_ps( _mm_mul_ps( *(__m128*)(blendMatrices+matrixIndex1+4) ,  weight ),  * m01 ); 
					* m02 = _mm_add_ps( _mm_mul_ps( *(__m128*)(blendMatrices+matrixIndex1+8) ,  weight ),  * m02 );
				}
		
	}



void Model::modifyVertexByJointKernelOptiSSE( float* pVertexArrayRaw , float* pVertexArrayDynamic ,int* pIndexJoint, float* pWeightJoint, Mesh* pMesh )
{
#if (ELEMENT_COUNT_POINT==4)

	__m128 *pSrcPos = (__m128*)(pVertexArrayRaw +STRIDE_POINT);
	__m128 *pDestPos = (__m128*)(pVertexArrayDynamic +STRIDE_POINT);

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
	for(int y = 0; y < pMesh->m_usNumTris*3; y++)
	{

#if (SIZE_PER_BONE==1)
		__m128 *pMatLast = (__m128*)(m_pJointsMatrix+ELEMENT_COUNT_MATIRX*pIndexJoint[y]);
#else
		__m128 pMatLast[4];
		
		__m128 *pMatOne = (__m128*)(m_pJointsMatrix+ELEMENT_COUNT_MATIRX*pIndexJoint[y*SIZE_PER_BONE+0]);
		float weight = pWeightJoint[y*SIZE_PER_BONE+0];
		__m128 scale = _mm_load_ps1( &weight );
		for(int i=0;i<4;i++)
		{
			pMatLast[i] = _mm_mul_ps(pMatOne[i], scale);
		}

		for (int i=1;i<SIZE_PER_BONE;i++)
		{
			pMatOne = (__m128*)(m_pJointsMatrix+ELEMENT_COUNT_MATIRX*pIndexJoint[y*SIZE_PER_BONE+i]);
			weight = pWeightJoint[y*SIZE_PER_BONE+i];
			scale = _mm_load_ps1( &weight );
			
			for (int j=0;j<4;j++)
			{
				pMatLast[j] = _mm_add_ps(pMatLast[j] ,_mm_mul_ps(pMatOne[j], scale) );
			}
		}
#endif

		__m128 vI0, vI1, vI2;

		vI0 = __MM_SELECT( pSrcPos[y], 0); 
		vI1 = __MM_SELECT( pSrcPos[y], 1); 
		vI2 = __MM_SELECT( pSrcPos[y], 2); 

		__m128 vO = __MM_DOT4x3_PS(pMatLast[0], pMatLast[1], pMatLast[2], pMatLast[3], vI0, vI1, vI2);  

		pDestPos[y] = vO;
	}

#else

	float *pDestPos = pVertexArrayDynamic + STRIDE_POINT;
	float *pSrcPos = pVertexArrayRaw +STRIDE_POINT;

	//遍历每个顶点
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
	for(int y = 0; y < pMesh->m_usNumTris*3; y++)
	{

		__m128 *pMatOne = (__m128*)(m_pJointsMatrix+ELEMENT_COUNT_MATIRX*pIndexJoint[y]);
		__m128 mat4[4];
		mat4[0] = pMatOne[0];
		mat4[1] = pMatOne[1];
		mat4[2] = pMatOne[2];
		mat4[3] = pMatOne[3];
		__MM_TRANSPOSE4x4_PS( mat4[2], mat4[3], mat4[0], mat4[1] );

		__m128 vI0, vI1, vI2;

		vI0 = _mm_load_ps1( pSrcPos + y*ELEMENT_COUNT_POINT );
		vI1 = _mm_load_ps1( pSrcPos + y*ELEMENT_COUNT_POINT + 1 );
		vI2 = _mm_load_ps1( pSrcPos + y*ELEMENT_COUNT_POINT + 2 );

		__m128 vO = __MM_DOT4x3_PS(mat4[2], mat4[3], mat4[0], mat4[1], vI0, vI1, vI2);  

		_mm_storeh_pi((__m64*)( pDestPos + y*ELEMENT_COUNT_POINT) , vO);
		_mm_store_ss( pDestPos + y * ELEMENT_COUNT_POINT + 2 , vO);

	}

#endif
}



void Model::SetupKernel(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue)
{
		int x = 1;
	
		float* pVertexArrayRaw = m_meshVertexData.m_pMesh[x].pVertexArrayRaw;
		float* pVertexArrayDynamic = m_meshVertexData.m_pMesh[x].pVertexArrayDynamic;

		int* pIndexJoint = m_meshVertexData.m_pMesh[x].pIndexJoint;
		float* pWeightJoint = m_meshVertexData.m_pMesh[x].pWeightJoint;

	_context = pContext;
	_device_ID = pDevice_ID;
	_kernel = pKernel;
	_cmd_queue = pCmdQueue;

	const cl_mem_flags INFlags  = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY; 
	const cl_mem_flags OUTFlags = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE;
	
	// allocate buffers
	g_pfInputBuffer		= clCreateBuffer(_context, INFlags,	sizeof(cl_float4) * ELEMENT_COUNT_POINT * m_pMeshes[1].m_usNumTris,	pVertexArrayRaw,	NULL);
	g_pfOCLOutputBuffer = clCreateBuffer(_context, OUTFlags,sizeof(cl_float4) * ELEMENT_COUNT_POINT * m_pMeshes[1].m_usNumTris ,	pVertexArrayDynamic,NULL);
	g_pfOCLIndex		= clCreateBuffer(_context, INFlags, sizeof(cl_int)	  * ELEMENT_COUNT_POINT * m_pMeshes[1].m_usNumTris ,	pIndexJoint,		NULL);   
	
	g_pfOCLMatrix		= clCreateBuffer(_context, INFlags, sizeof(cl_float4) * MATRIX_SIZE_LINE	* m_usNumJoints ,		m_pJointsMatrix,	NULL); 


	//Set kernel arguments
	clSetKernelArg(_kernel, 0, sizeof(cl_mem), (void *) &g_pfInputBuffer);
	clSetKernelArg(_kernel, 1, sizeof(cl_mem), (void *) &g_pfOCLIndex);
	clSetKernelArg(_kernel, 2, sizeof(cl_mem), (void *) &g_pfOCLMatrix);
	clSetKernelArg(_kernel, 3, sizeof(cl_mem), (void *) &g_pfOCLOutputBuffer);
}

void Model::SetupWorksize( size_t* globalWorkSize, size_t* localWorkSize, int dim )
{
	globalWorkSize[0] = (size_t)sqrtf( ELEMENT_COUNT_POINT * m_pMeshes[1].m_usNumTris );
	globalWorkSize[1] = globalWorkSize[0];

	globalWorkSize[0]/=4; //since proccesing in quadruples


	localWorkSize[0] = LocalWorkX;
	localWorkSize[1] = LocalWorkX;
	printf("Original global work size (%lu, %lu)\n", globalWorkSize[0], globalWorkSize[1]);
	printf("Original local work size (%lu, %lu)\n", localWorkSize[0], localWorkSize[1]);

	size_t  workGroupSizeMaximum;
	clGetKernelWorkGroupInfo(_kernel, _device_ID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *)&workGroupSizeMaximum, NULL);
	printf("Maximum workgroup size for this kernel  %lu\n\n",workGroupSizeMaximum );

	if ( m_pMeshes[1].m_usNumTris > workGroupSizeMaximum )
	{
		globalWorkSize[0] = workGroupSizeMaximum;
		globalWorkSize[1] = m_pMeshes[1].m_usNumTris / workGroupSizeMaximum;
	}
	printf("Actual global work size (%lu, %lu)\n", globalWorkSize[0], globalWorkSize[1]);
}

bool Model::ExecuteKernel(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue)
{
	cl_int err = CL_SUCCESS;

	SetupKernel(pContext, pDevice_ID, pKernel, pCmdQueue);


	size_t globalWorkSize[2];
	size_t localWorkSize[2];
	SetupWorksize(globalWorkSize, localWorkSize, 2);

	printf("Executing OpenCL kernel...");

	cl_event g_perf_event = NULL;
	// execute kernel, pls notice g_bAutoGroupSize
	err= clEnqueueNDRangeKernel(_cmd_queue, _kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &g_perf_event);
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

	printf("Done\n");

	float* pVertexArrayDynamic = m_meshVertexData.m_pMesh[1].pVertexArrayDynamic;
	void* tmp_ptr = NULL;
	err = clEnqueueReadBuffer(_cmd_queue, g_pfOCLOutputBuffer, CL_TRUE, 0, sizeof(cl_float4) * ELEMENT_COUNT_POINT * m_pMeshes[1].m_usNumTris , pVertexArrayDynamic, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("ERROR: Failed to clEnqueueReadBuffer...\n");
		return false;
	}

	clFinish(_cmd_queue);

	clEnqueueUnmapMemObject(_cmd_queue, g_pfOCLOutputBuffer, tmp_ptr, 0, NULL, NULL);

	return true;
}
