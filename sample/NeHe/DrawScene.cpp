#include "DrawScene.h"


int DrawScene::DrawGLScene( )
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);				// Clear The Screen And The Depth Buffer
	glLoadIdentity();												// Reset The Modelview Matrix
	gluLookAt( 75, 75, 75, 0, 0, 0, 0, 1, 0 );						// (3) Eye Postion (3) Center Point (3) Y-Axis Up Vector

	glRotatef(yrot,0.0f,1.0f,0.0f);									// Rotate On The Y-Axis By yrot

	long timerBeginMiliSecond = clock();
	m_model.draw();													// Draw The Model
	long timerEndMiliSecond = clock();

	long timeElapsed = timerEndMiliSecond - timerBeginMiliSecond;
	//cout << timeElapsed << endl;

	if ( timeElapsed > 1 )
	{
		//_cprintf("‰÷»æ∫ƒ ±Time Elapsed%d", timeElapsed );
		printf("‰÷»æ∫ƒ ±Time Elapsed %d \n", timeElapsed );
	}

	yrot+=1.0f;														// Increase yrot By One
	return TRUE;
}

int DrawScene::InitGL( )
{
	AllocConsole(); 
	freopen( "CONOUT$","w",stdout);


	m_model.reloadTextures();										// Loads Model Textures

	glEnable(GL_TEXTURE_2D);										// Enable Texture Mapping ( NEW )
	glShadeModel(GL_SMOOTH);										// Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);							// Black Background
	glClearDepth(1.0f);												// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);										// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);											// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);				// Really Nice Perspective Calculations
	return TRUE;
}

bool DrawScene::loadModelData( string filename )
{
	if ( m_model.loadModelData( filename.c_str() ) == false )		// Loads The Model And Checks For Errors
	{
		MessageBox( NULL, "Couldn't load the model data\\model.ms3d", "Error", MB_OK | MB_ICONERROR );
		return 0;													// If Model Didn't Load Quit
	}
}
