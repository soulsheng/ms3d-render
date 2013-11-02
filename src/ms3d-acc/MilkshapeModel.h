/*
	MilkshapeModel.h

		Loads and renders a Milkshape3D model. 

	Author:	Brett Porter
	Email: brettporter@yahoo.com
	Website: http://www.geocities.com/brettporter/
	Copyright (C)2000, Brett Porter. All Rights Reserved.

	This file may be used only as long as this copyright notice remains intact.
*/

#ifndef MILKSHAPEMODEL_H
#define MILKSHAPEMODEL_H

#include "Model.h"


class MilkshapeModel : public Model
{
	public:
		/*	Constructor. */
		MilkshapeModel();

		/*	Destructor. */
		virtual ~MilkshapeModel();

		/*	
			Load the model data into the private variables. 
				filename			Model filename
		*/
		virtual bool loadModelData( const char *filename );

		
		/*
			Draw the model.
		*/
		virtual void draw();
		
		virtual void Setup();

		virtual bool ExecuteKernel(cl_context pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue);

		virtual void SetupKernel(cl_context pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue);
		
		void SetupGLSL();
		void clearGLSL();

	protected:
		void initializeVBO();

		void renderVBO();

		void modifyVBO();

		void modifyVBOOpti();

protected:
	void PreSetup();

protected:	

		//unsigned int maxMeshVertexNumber;

		GLuint _idVBOFaceIndexAll;

		unsigned int* _idGPURenderItemsPerMesh;

		// glsl
		GLuint vertexShader, pixelShader;
		GLuint glProgram;

		int   _locationUniformMatrix;		// 参数位置，矩阵数据
		int   _locationUniformMultiBone;	// 参数位置，每个顶点绑定的骨骼数

		int   _locationAttributeIndex;		// 参数位置，矩阵索引
		int   _locationAttributeWeight;		// 参数位置，矩阵权重
};

#endif // ndef MILKSHAPEMODEL_H
