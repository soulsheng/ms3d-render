

#pragma once



#include "MilkshapeModel.h"	
										// Header File For Milkshape File
class DrawScene
{
public:
	int InitGL( )	;												// All Setup For OpenGL Goes Here

	bool loadModelData(string filename);

	int DrawGLScene()	;

protected:
private:
	MilkshapeModel m_model;
	GLfloat		yrot;
};
