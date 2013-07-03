/*************************************************************/
/*                        VGCATIMER.H                        */
/*                                                           */
/* Purpose: Definitions and Implementation for the Timer    */
/*          class and relatedfunctions used for timing       */
/*          purposes.                                        */
/*      Evan Pipho (evan@codershq.com)                       */
/*                                                           */
/*************************************************************/
#ifndef VGKTIMER_H
#define VGKTIMER_H

//#include "vgLibraryPrerequisites.h"
//#include "vgkForward.h"

#include <windows.h>


/** 
* @file vgKernel/include/vgkTimer.h
* @brief 系统时间相关功能.
* @author Evan Pipho (evan@codershq.com) 
* @date Jul 09, 2002 
* @version 3.0
*/

namespace vgKernel {

	/**
	* @class vgKernel::Timer
	* @brief 时间类. 
	*/
	class	 Timer
	{
	public:

		/**
		* @brief Initialize the timer
		*/
		void Init();

		/**
		* @brief Get the FPS of the program baed on elapsed frames
		* @return FPS result, estimate the render times per second
		*/
		float GetFPS(unsigned int uiFrames = 1);

		/**
		* @brief Get the number of ms since last call
		* @return milliseconds past
		*/
		unsigned int GetMS();
		
		/**
		* @brief Get the number of seconds since the last call
		* @return seconds past
		*/
		float GetSeconds();

	private:

		//multipling by this gives you time in seconds	
		float m_fInvTicksPerSec;	/**< 单位因子，乘以该值返回秒 */
		//multipling by this gives you time in milliseconds
		float m_fInvTicksPerMs;		/**< 单位因子，乘以该值返回豪秒 */
		//start time
		__int64 m_i64StartTime;
		__int64 m_i64LastTime;
	};

	//-------------------------------------------------------------
	//                     FUNCTIONS                              -
	//-------------------------------------------------------------
	//-------------------------------------------------------------
	//- Init
	//- Initialize the timer
	//-------------------------------------------------------------
	inline void Timer::Init()
	{
		__int64 i64Freq = 0;
		BOOL ret = QueryPerformanceFrequency((LARGE_INTEGER *)&i64Freq);
		//assert( ret == TRUE );
		ret = QueryPerformanceCounter((LARGE_INTEGER *)&m_i64StartTime);
		//assert( ret == TRUE );

		m_fInvTicksPerSec = (float)1/i64Freq;
		m_fInvTicksPerMs = (float)1000/i64Freq;
		m_i64LastTime = 0;
		GetMS();
	}

	//-------------------------------------------------------------
	//- GetFPS
	//- Calulate the Frames Per Second of the application
	//-------------------------------------------------------------
	inline float Timer::GetFPS(unsigned int uiFrames)
	{
		static __int64 sLastTime = m_i64StartTime;
		__int64 i64NewTime = 0;
		float fElapsedSecs = 0;
		QueryPerformanceCounter((LARGE_INTEGER *)&i64NewTime);
		fElapsedSecs = (i64NewTime - sLastTime) * m_fInvTicksPerSec;
		sLastTime = i64NewTime;

		return (uiFrames / fElapsedSecs); 
	}

	//-------------------------------------------------------------
	//- GetMS
	//- Get the number of milliseconds since the last call to GetMS
	//- or since the start of the timer.
	//-------------------------------------------------------------
	inline unsigned int Timer::GetMS()
	{
		__int64 i64NewTime = 0;
		unsigned int uiMs;

		QueryPerformanceCounter((LARGE_INTEGER *)&i64NewTime);
		uiMs = (unsigned int)((i64NewTime - m_i64LastTime) * m_fInvTicksPerMs);
		m_i64LastTime = i64NewTime;
		return uiMs;
	}

	//-------------------------------------------------------------
	//- GetSeconds
	//- Get the number of seconds since the last call to GetSeconds
	//-------------------------------------------------------------
	inline float Timer::GetSeconds()
	{
		return (float)(GetMS() * 0.001f);
	}

}

#endif //VGKTIMER_H