

#pragma once

#include <windows.h>												// Header File For Windows
#include <stdio.h>													// Header File For Standard Input/Output
#include <gl\gl.h>													// Header File For The OpenGL32 Library
#include <gl\glu.h>													// Header File For The GLu32 Library
#include <gl\glaux.h>												// Header File For The Glaux Library

#include <time.h>
#include <conio.h>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;

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
