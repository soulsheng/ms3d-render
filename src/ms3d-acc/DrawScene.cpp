#include "stdafx.h"
#include "DrawScene.h"

#define FILENAME_MS3D "data/tortoise.ms3d"
//#define FILENAME_MS3D "data/Dophi.ms3d"

int DrawScene::DrawGLScene( )
{

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);				// Clear The Screen And The Depth Buffer
	glLoadIdentity();												// Reset The Modelview Matrix
	gluLookAt( 75, 75, 575, 0, 0, 0, 0, 1, 0 );						// (3) Eye Postion (3) Center Point (3) Y-Axis Up Vector

	glRotatef(yrot,0.0f,1.0f,0.0f);									// Rotate On The Y-Axis By yrot

	//long timerBeginMiliSecond = clock();
#if ENABLE_DRAW_REPEAT
	// 每个动画包含顶点个数是24K，总顶点个数是1M
	for(int nIndex = 0; nIndex < COUNT_MODEL; nIndex++ )
	{
		m_model->draw();													// Draw The Model
	}
#else
	for(int nIndex = 0; nIndex < COUNT_MODEL; nIndex++ )
	{
		m_model[nIndex].draw();													// Draw The Model
	}
#endif
	//long timerEndMiliSecond = clock();

	//long timeElapsed = timerEndMiliSecond - timerBeginMiliSecond;

	//cout << "渲染耗时Time Elapsed " << timeElapsed << endl;
	
	glFinish();

#if ENABLE_TIMER
	_timer.stopTimer(timer1);
	double dTime1 = (double)_timer.readTimer(timer1);

#if ENABLE_FPS_COUNT
	static double dSecond = 0.0f;
	static int	nFPS = 0;
	if ( dSecond<1.0f )
	{
		dSecond += dTime1;
		nFPS ++ ;
	}
	else
	{
		_timer.insertTimer("nFPS", nFPS);

		dSecond = 0.0f;
		nFPS = 0;
	}
#else
	_timer.insertTimer("runtime", dTime1);
#endif

	_timer.resetTimer(timer1);
	_timer.startTimer(timer1);
#endif
	return TRUE;
}

int DrawScene::InitGL( )
{

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		::MessageBox(NULL,"glewInit failed, something is seriously wrong.",
			"glew error occured.",MB_OK );
		return false;
	}
#if CUDA_ENABLE
	// cuda初始化
	cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
#endif

	for (int i=0;i<COUNT_MODEL;i++)
	{
		m_model[i].loadModelData(FILENAME_MS3D);
		m_model[i].reloadTextures();										// Loads Model Textures
	}

	glEnable(GL_TEXTURE_2D);										// Enable Texture Mapping ( NEW )
	glShadeModel(GL_SMOOTH);										// Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);							// Black Background
	glClearDepth(1.0f);												// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);										// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);											// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);				// Really Nice Perspective Calculations
#if ENABLE_CONSOLE_WINDOW
	AllocConsole(); 
	freopen( "CONOUT$","w",stdout);
#endif
	return TRUE;
}

void DrawScene::printfTimer()
{
	std::ostringstream oss;
	_timer.printfTimer( oss );
	std::cout << oss.str();
}

DrawScene::DrawScene()
{
#if ENABLE_TIMER
	timer1 = _timer.createTimer();

	_timer.resetTimer(timer1);
	_timer.startTimer(timer1);
#endif
	m_model = new MilkshapeModel[COUNT_MODEL];
}

DrawScene::~DrawScene()
{
	if ( m_model )
	{
		delete[] m_model;
		m_model = NULL;
	}
}

