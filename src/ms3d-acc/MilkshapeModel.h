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

// CUDA runtime
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <cuda_gl_interop.h>
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

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

		void initializeCUDA( );

	protected:
		void initializeVBO();

		void renderVBO();

		void modifyVBO();

		void modifyVBOOpti();

protected:
	void PreSetup();
	bool runCUDAHost();
protected:	

		//unsigned int maxMeshVertexNumber;

		GLuint _idVBOFaceIndexAll;

		unsigned int* _idGPURenderItemsPerMesh;

		struct  CUDAKernelArguments
		{
			float	*d_pInput, *d_pOutput ;  // æÿ’Û±‰ªª ‰»Î ‰≥ˆ∂•µ„
			float	*d_pMatrix, *d_pWeight ; // æÿ’Û£¨ æÿ’Û»®÷ÿ
			int		*d_pIndex;				 // æÿ’ÛÀ˜“˝
			CUDAKernelArguments()
			{
				d_pInput = d_pOutput = d_pMatrix = d_pWeight = NULL;
				d_pIndex = NULL;
			}

			~CUDAKernelArguments()
			{
				if (d_pInput) cudaFree(d_pInput) ;
				if (d_pOutput) cudaFree(d_pOutput) ;
				if (d_pMatrix) cudaFree(d_pMatrix) ;
				if (d_pWeight) cudaFree(d_pWeight) ;
				if (d_pIndex) cudaFree(d_pIndex) ;
			}
		};

		CUDAKernelArguments	_cudaKernelArguments;
		struct cudaGraphicsResource *cuda_vbo_resource;
};

#endif // ndef MILKSHAPEMODEL_H
