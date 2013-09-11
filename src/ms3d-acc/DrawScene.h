

#pragma once



#include "MilkshapeModel.h"	
										// Header File For Milkshape File
#include "timerOMP.h"
#include "timer.h"

class DrawScene
{
public:
	int InitGL( )	;												// All Setup For OpenGL Goes Here

	int DrawGLScene()	;

	void printfTimer();

	DrawScene();
	~DrawScene();

protected:
private:
	MilkshapeModel* m_model;
	GLfloat		yrot;

	CTimer	_timer;
	int timer1;
};
