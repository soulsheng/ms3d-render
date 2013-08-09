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
			glInterleavedArrays(GL_T2F_N3F_V3F, 0, m_meshVertexData.m_pMesh[i].pVertexArrayDynamic);

#if RENDERMODE_POINT
			glDrawArrays(GL_POINTS, 0, m_pMeshes[i].m_usNumTris * 3 );			
#else
			glDrawElements(GL_TRIANGLES, m_meshVertexData.m_pMesh[i].numOfVertex,
				GL_UNSIGNED_INT , m_pIndexArray);
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
	{//�޸ĵĲ���
		fLastTime= fStartTime;
		m_Timer.Init();
		m_Timer.GetSeconds();
		bFirstTime = false;
	}

	if (m_bPlay)
	{
		fTime = m_Timer.GetSeconds() * fSpeed;//֡ʱ��
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
	
	// ����ÿ��Mesh������Joint����ÿ��Vertex������
	for(int x = 0; x < m_usNumMeshes; x++)
	{
		float* pVertexArrayStatic = m_meshVertexData.m_pMesh[x].pVertexArrayStatic;
		float* pVertexArrayDynamic = m_meshVertexData.m_pMesh[x].pVertexArrayDynamic;
		float* pVertexArrayRaw = m_meshVertexData.m_pMesh[x].pVertexArrayRaw;

		int* pIndexJoint = m_meshVertexData.m_pMesh[x].pIndexJoint;
		
		Mesh* pMesh = m_pMeshes+x;
		
		modifyVertexByJointInitKernel(  pVertexArrayRaw, pVertexArrayStatic, pIndexJoint, pMesh ); // ����pVertexArrayStatic

#if !RENDERMODE_MOVING
		memcpy( pVertexArrayDynamic, pVertexArrayStatic,  (2+3+3) * m_pMeshes[x].m_usNumTris * 3 * sizeof(float) );
#endif
	}
}

void Model::modifyVertexByJoint()
{
	
	// ����ÿ��Mesh������Joint����ÿ��Vertex������
	//int x = 2;
	for(int x = 0; x < m_usNumMeshes; x++)
	{
		float* pVertexArrayRaw = m_meshVertexData.m_pMesh[x].pVertexArrayRaw;
		float* pVertexArrayDynamic = m_meshVertexData.m_pMesh[x].pVertexArrayDynamic;

		int* pIndexJoint = m_meshVertexData.m_pMesh[x].pIndexJoint;
		
		Mesh* pMesh = m_pMeshes+x;
		
#if ENABLE_OPTIMIZE
		modifyVertexByJointKernelOpti(  pVertexArrayRaw, pVertexArrayDynamic, pIndexJoint, pMesh );
#else
		modifyVertexByJointKernel(  pVertexArrayDynamic, pIndexJoint, pMesh );
#endif
	}
}

void Model::setupVertexArray()
{
	// ����Mesh������m_usNumMeshes��VertexArray.
	// ��m_meshVertexData�������������ͷ�.
	m_meshVertexData.m_pMesh = new Ms3dVertexArrayMesh[m_usNumMeshes];
	m_meshVertexData.m_numberOfMesh = m_usNumMeshes;


	for(int x = 0; x < m_usNumMeshes; x++)
	{
		// ��m_pMesh�������������ͷ�. ����Ҫ���ӷ�����ʹ��
		float* pVertexArrayStatic = new float[(2+3+3) * m_pMeshes[x].m_usNumTris * 3];
		float* pVertexArrayDynamic = new float[(2+3+3) * m_pMeshes[x].m_usNumTris * 3];
		int* pIndexJoint = new int[m_pMeshes[x].m_usNumTris * 3];
		float* pVertexArrayRaw = new float[(2+3+3) * m_pMeshes[x].m_usNumTris * 3];

		m_meshVertexData.m_pMesh[x].numOfVertex = m_pMeshes[x].m_usNumTris * 3;
		m_meshVertexData.m_pMesh[x].pVertexArrayStatic = pVertexArrayStatic;
		m_meshVertexData.m_pMesh[x].pVertexArrayDynamic = pVertexArrayDynamic;
		m_meshVertexData.m_pMesh[x].pIndexJoint = pIndexJoint;
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

				// ����ʼ������
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

	//����Mesh��ÿ��������
	for(int y = 0; y < pMesh->m_usNumTris; y++)
	{
		//Set triangle pointer to triangle #1

		Triangle * pTri = &m_pTriangles[pMesh->m_uspIndices[y]];
		// ������������������� 
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
				// ��ǰ�汾�����㷨��					
				// vecNormal.Transform3(pJoint->m_matFinal);

				// Transform the vertex
				vecVertex = pVert->m_vVert;
				// translate as well as rotate
				vecVertex.Transform4(pJoint->m_matFinal);

			}

			vertexCnt += 2;

			// ����û�б�����Ϳ���
			pVertexArrayDynamic[vertexCnt++] = vecNormal[0];
			pVertexArrayDynamic[vertexCnt++] = vecNormal[1];
			pVertexArrayDynamic[vertexCnt++] = vecNormal[2];

			pVertexArrayDynamic[vertexCnt++] = vecVertex[0];
			pVertexArrayDynamic[vertexCnt++] = vecVertex[1];
			pVertexArrayDynamic[vertexCnt++] = vecVertex[2];
		}//for z
	}//for y
}

void Model::modifyVertexByJointInitKernel( float* pVertexArrayStatic , float* pVertexArrayDynamic  , int* pIndexJoint, Mesh* pMesh)
{
	vgMs3d::CVector3 vecNormal;
	vgMs3d::CVector3 vecVertex;

	int vertexCnt = 0;

	//����Mesh��ÿ��������
	for(int y = 0; y < pMesh->m_usNumTris; y++)
	{
		//Set triangle pointer to triangle #1

		Triangle * pTri = &m_pTriangles[pMesh->m_uspIndices[y]];
		// ������������������� 
		for(int z = 0; z < 3; z++)
		{
			//Get the vertex
			Vertex * pVert = &m_pVertices[pTri->m_usVertIndices[z]];
			
			pIndexJoint[3*y+z] = pVert->m_cBone;

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

			vertexCnt += 2;

			// ����û�б�����Ϳ���
			pVertexArrayDynamic[vertexCnt++] = vecNormal[0];
			pVertexArrayDynamic[vertexCnt++] = vecNormal[1];
			pVertexArrayDynamic[vertexCnt++] = vecNormal[2];

			pVertexArrayDynamic[vertexCnt++] = vecVertex[0];
			pVertexArrayDynamic[vertexCnt++] = vecVertex[1];
			pVertexArrayDynamic[vertexCnt++] = vecVertex[2];

			if(pVertexArrayStatic)
			{
				pVertexArrayStatic[ 8*(3*y+z) + 5 ] = pVert->m_vVert[0];
				pVertexArrayStatic[ 8*(3*y+z) + 6 ] = pVert->m_vVert[1];
				pVertexArrayStatic[ 8*(3*y+z) + 7 ] = pVert->m_vVert[2];
			}
		}//for z
	}//for y
}


void Model::modifyVertexByJointKernelOpti( float* pVertexArrayRaw , float* pVertexArrayDynamic , int* pIndexJoint, Mesh* pMesh)
{
	int vertexCnt = 0;

	//����ÿ������
	for(int y = 0; y < pMesh->m_usNumTris*3; y++)
	{
#if 0
		float *sourceVec = pVertexArrayRaw + 8*y+5;
		float *desVec = pVertexArrayDynamic + 8*y+5;

		vgMs3d::CVector3 vecVertex(sourceVec);
		
		vecVertex.Transform4( m_pJoints[ pIndexJoint[y] ].m_matFinal );

		desVec[ 0 ] = vecVertex[0];
		desVec[ 1 ] = vecVertex[1];
		desVec[ 2 ] = vecVertex[2];
#else
		float *sourceVec = pVertexArrayRaw + 8*y+5;
		float *desVec = pVertexArrayDynamic + 8*y+5;

		const float* mat = m_pJoints[ pIndexJoint[y] ].m_matFinal.Get();
		
		desVec[0] =
			(mat[0*4+0] * sourceVec[0] +
			mat[1*4+0] * sourceVec[1] +
			mat[2*4+0] * sourceVec[2] +
			mat[3*4+0]) ;
		
		desVec[1] =
			(mat[0*4+1] * sourceVec[0] +
			mat[1*4+1] * sourceVec[1] +
			mat[2*4+1] * sourceVec[2] +
			mat[3*4+1]) ;

		desVec[2] =
			(mat[0*4+2] * sourceVec[0] +
			mat[1*4+2] * sourceVec[1] +
			mat[2*4+2] * sourceVec[2] +
			mat[3*4+2]) ;

#endif
	}//for y
}

