
#include "stdafx.h"
#include "timerOMP.h"

#include <windows.h>
#include <omp.h>

#define SDK_SUCCESS  0
#define SDK_FAILURE  -1


int CTimerOMP::createTimer()
{
    _start = 0;
	_end = 0;
	
	return SDK_SUCCESS;
}

int CTimerOMP::resetTimer(int handle)
{
	_start = 0;
	_end = 0;

    return SDK_SUCCESS;
}

int CTimerOMP::startTimer(int handle)
{
	warmup();

	_start = omp_get_wtime();

    return SDK_SUCCESS;
}

int CTimerOMP::stopTimer(int handle)
{
	_end= omp_get_wtime();

	SetThreadAffinityMask(::GetCurrentThread(), oldmask);

    return SDK_SUCCESS;
}

double CTimerOMP::readTimer(int handle)
{

    return _end - _start;
}


void CTimerOMP::printfTimer( std::ostringstream &oss )
{
	double dAverageTime = 0.0f;
	double dMeanSquareError = 0.0f;
	int    nItem = 0;
	double dMax = 0.0f;
	double dMin = 100.0f;
	std::vector<double> dValidTimeVec;
	double dLast = 100.0f;
	for (TimerValueListItr itr=_timeValueList.begin(); itr!=_timeValueList.end(); itr++,nItem++)
	{
		std::cout << itr->first << ":  " << itr->second << std::endl;
		if ( nItem >= 20 )
		{
			if ( 1.5f*dLast < itr->second )
			{
				continue;
			}
			dLast = itr->second;
			dValidTimeVec.push_back(itr->second*1000);

			if ( dMax < itr->second*1000 )
			{
				dMax = itr->second*1000 ;
			}

			if ( dMin > itr->second*1000 )
			{
				dMin = itr->second*1000 ;
			}
		}
	}

	for (int i=0; i<dValidTimeVec.size(); i++)
	{
		dAverageTime += dValidTimeVec[i];
	}
	dAverageTime /= dValidTimeVec.size();

	for (int i=0; i<dValidTimeVec.size(); i++)
	{
		dMeanSquareError += (dValidTimeVec[i] - dAverageTime)*(dValidTimeVec[i] - dAverageTime);
	}

	oss << "AverageTime is: " << std::setprecision(3) << dAverageTime << std::endl;
	oss << "MaxTime is: " << dMax  << ", MinTime is: " << dMin<< std::endl ;
	oss << "MeanSquareError is: " << sqrtl(dMeanSquareError/dValidTimeVec.size() ) << std::endl ;
	oss << "MeanSquareErrorRatio is: " << sqrtl(dMeanSquareError/dValidTimeVec.size() )/dAverageTime << std::endl ;
	oss << std::endl;
}

void CTimerOMP::insertTimer( std::string timeString, double timeValue)
{
	if ( _timeValueList.size()>200 )
	{
		return;
	}
	_timeValueList.insert( std::make_pair(timeString, timeValue) );
}

void CTimerOMP::clear()
{
	_timeValueList.clear();
}

void CTimerOMP::warmup()
{
	volatile int warmingUp = 1;
#pragma omp parallel for
	for (int i=1; i<10000000; i++)
	{
#pragma omp atomic
		warmingUp *= i;
	}

	oldmask = SetThreadAffinityMask(::GetCurrentThread(), 1);
}
