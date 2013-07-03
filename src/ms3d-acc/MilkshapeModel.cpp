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


#define BUFFER_OFFSET(i) ((char *)NULL + (i))

MilkshapeModel::MilkshapeModel()
{
}

MilkshapeModel::~MilkshapeModel()
{
	delete[] _metaFaces;

#if CUDA_ENABLE
	// cuda 销毁
	cudaThreadExit();
#endif
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
	for ( i = 0; i < nGroups; i++ )
	{
		pPtr += sizeof( char );	// flags
		pPtr += 32;				// name

		word nTriangles = *( word* )pPtr;
		pPtr += sizeof( word );
		int *pTriangleIndices = new int[nTriangles];
		for ( int j = 0; j < nTriangles; j++ )
		{
			pTriangleIndices[j] = *( word* )pPtr;
			pPtr += sizeof( word );
		}

		char materialIndex = *( char* )pPtr;
		pPtr += sizeof( char );
	
		m_pMeshes[i].m_materialIndex = materialIndex;
		m_pMeshes[i].m_usNumTris = nTriangles;
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

#if RENDERMODE_VBO
	initializeVBO();
#endif


	delete[] pBuffer;

	Setup();

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

	// 计算点UV和面索引
	_metaFaces = new VboMetaFaceStruct[m_usNumMeshes];
	for ( int i = 0; i < m_usNumMeshes ; ++i )
	{
		Mesh& mesh = m_pMeshes[i];
		int nSizeTriangles = mesh.m_usNumTris;

		_metaFaces[i]._numOfElements = nSizeTriangles;
		_metaFaces[i].pFaceIndex = new VboFaceIndex[nSizeTriangles];
		for (int nIndexTrianglesIndex = 0; nIndexTrianglesIndex < nSizeTriangles; nIndexTrianglesIndex++ )
		{
			int nIndexTriangle = mesh.m_uspIndices[nIndexTrianglesIndex];
			Triangle& tri = m_pTriangles[nIndexTriangle];
			memcpy( &_metaFaces[i].pFaceIndex[nIndexTrianglesIndex], &tri.m_usVertIndices, sizeof( int )*3 );

			// uv
			for (int index = 0;index <3; index++)
			{
				int nIndexVertex = tri.m_usVertIndices[index];
				m_pVertices[nIndexVertex].m_texcoord[0] = tri.m_s[index];
				m_pVertices[nIndexVertex].m_texcoord[1] = tri.m_t[index];
			}
		}
	}

	initializeVBOAttribute();

	initializeVBOMesh();	
}

void MilkshapeModel::renderVBO()
{
	glBindBufferARB( GL_ARRAY_BUFFER_ARB, _idVBOVertexArray );

	glColor3f(1.0f, 1.0f, 1.0f);
	glPointSize(2.0f);
	
	glVertexPointer( 3, GL_FLOAT, sizeof(Vertex), BUFFER_OFFSET( sizeof(char)  ) );
	glTexCoordPointer( 2, GL_FLOAT, sizeof(Vertex), BUFFER_OFFSET( sizeof(char) + sizeof(float)*3 ) );
	
	glEnableClientState( GL_TEXTURE_COORD_ARRAY );
	glEnableClientState( GL_VERTEX_ARRAY );

#if RENDERMODE_POINT
	glDrawArrays(GL_POINTS, 0, m_usNumVerts );
#else
	for ( int i = 0; i < m_usNumMeshes ; ++i )
	{
		glBindBufferARB( GL_ELEMENT_ARRAY_BUFFER_ARB, _metaFaces[i]._elementBufferObjectID );

		glDrawElements( GL_TRIANGLES, _metaFaces[i]._numOfElements*3, GL_UNSIGNED_INT, BUFFER_OFFSET(0));

		glBindBufferARB( GL_ELEMENT_ARRAY_BUFFER_ARB, NULL );
	}
#endif

	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );

	glBindBufferARB( GL_ARRAY_BUFFER_ARB, NULL );

}

void MilkshapeModel::modifyVBO()
{

}

void MilkshapeModel::createVBO( GLuint* vbo, unsigned int size, void* pData, int type /*= GL_ARRAY_BUFFER_ARB*/, int usage /*= GL_STATIC_DRAW*/ )
{
	glGenBuffersARB(1, vbo);
	glBindBufferARB(type, *vbo);

	// initialize buffer object
	glBufferDataARB(type, size, pData, usage);

	glBindBufferARB(type, 0);
}

void MilkshapeModel::initializeVBOAttribute()
{
	


}

void MilkshapeModel::initializeVBOMesh()
{
	// 点
	unsigned int sizeVertex = sizeof(Vertex) * m_usNumVerts;
	createVBO( &_idVBOVertexArray , sizeVertex , (void*)m_pVertices );

	// 面
	for ( int i = 0; i < m_usNumMeshes ; ++i )
	{
		unsigned int sizeFaceIndex = sizeof(VboFaceIndex) * _metaFaces[i]._numOfElements;
		createVBO( &_metaFaces[i]._elementBufferObjectID , sizeFaceIndex , _metaFaces[i].pFaceIndex, GL_ELEMENT_ARRAY_BUFFER_ARB);
		delete[]  _metaFaces[i].pFaceIndex;
	}
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

	modifyVertexByJoint();

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


VboMetaFaceStruct::VboMetaFaceStruct()
{

}

VboMetaFaceStruct::~VboMetaFaceStruct()
{

}
