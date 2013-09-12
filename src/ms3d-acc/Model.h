/*
	Model.h

		Abstract base class for a model. The specific extended class will render the given model. 

	Author:	Brett Porter
	Email: brettporter@yahoo.com
	Website: http://www.geocities.com/brettporter/
	Copyright (C)2000, Brett Porter. All Rights Reserved.

	This file may be used only as long as this copyright notice remains intact.
*/

#ifndef MODEL_H
#define MODEL_H

#include "StructMS3D.h"
#include "math/vgkTimer.h"

class Model
{
		

	public:
		/*	Constructor. */
		Model();

		/*	Destructor. */
		virtual ~Model();

		/*	
			Load the model data into the private variables. 
				filename			Model filename
		*/
		virtual bool loadModelData( const char *filename ) = 0;

		/*
			Draw the model.
		*/
		virtual void draw();

		virtual void Setup() = 0;

		/*
			Called if OpenGL context was lost and we need to reload textures, display lists, etc.
		*/
		void reloadTextures();

protected:

	void updateJoints(float fTime);

	void modifyVertexByJointInit();
	void modifyVertexByJointInitKernel( float* pVertexArrayStatic , float* pVertexArrayDynamic , int* pIndexJoint, Mesh* pMesh );

	void modifyVertexByJoint();
	void modifyVertexByJointKernel( float* pVertexArrayDynamic , int* pIndexJoint, Mesh* pMesh );
	void modifyVertexByJointKernelOpti( float* pVertexArrayRaw , float* pVertexArrayDynamic ,int* pIndexJoint,  Mesh* pMesh );

	void getPlayTime(float fSpeed, float fStartTime, float fEndTime, bool bLoop);
	
	void setupVertexArray();

	protected:
		//	Meshes used
		int m_usNumMeshes;
		Mesh *m_pMeshes;

		//	Materials used
		int m_numMaterials;
		Material *m_pMaterials;

		//	Triangles used
		int m_usNumTriangles;
		Triangle *m_pTriangles;

		//	Vertices Used
		int m_usNumVerts;
		Vertex *m_pVertices;

		//	Joints used
		int m_usNumJoints;
		MS3DJoint*	m_pJoints;
		float*		m_pJointsMatrix;

		//Total time for model animation
		float m_fTotalTime;

		Ms3dIntervelData m_meshVertexData;
		unsigned int *m_pIndexArray;
		unsigned int maxMeshVertexNumber;

protected:
	bool m_load;
	bool bFirstTime;
	float fLastTime;
	float fTime;
	vgKernel::Timer m_Timer;
	
	bool m_bPlay, b_loop;

protected:
	//Draw the bones?
	bool m_bDrawBones;
	//Draw the mesh?
	bool m_bDrawMesh;

};

#endif // ndef MODEL_H
