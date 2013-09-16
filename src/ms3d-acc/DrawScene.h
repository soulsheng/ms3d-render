

#pragma once



										// Header File For Milkshape File
#include "timerOMP.h"
#include "timer.h"
#include "COclManager.h"

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
	GLfloat		yrot;

	CTimer	_timer;
	int timer1;

	COclManager		oclManager;
};
