
#include "stdafx.h"
#include "timer.h"

#include <windows.h>

#define SDK_SUCCESS  0
#define SDK_FAILURE  -1


int CTimer::createTimer()
{
    Timer* newTimer = new Timer;
    newTimer->_start = 0;
    newTimer->_clocks = 0;

#ifdef _WIN32
    QueryPerformanceFrequency((LARGE_INTEGER*)&newTimer->_freq);
#else
    newTimer->_freq = (long long)1.0E3;
#endif
    
    /* Push back the address of new Timer instance created */
    _timers.push_back(newTimer);

    return (int)(_timers.size() - 1);
}

int CTimer::resetTimer(int handle)
{
    if(handle >= (int)_timers.size())
    {
        error("Cannot reset timer. Invalid handle.");
        return -1;
    }

    (_timers[handle]->_start) = 0;
    (_timers[handle]->_clocks) = 0;
    return SDK_SUCCESS;
}

int CTimer::startTimer(int handle)
{
    if(handle >= (int)_timers.size())
    {
        error("Cannot reset timer. Invalid handle.");
        return SDK_FAILURE;
    }

	warmup();

#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER*)&(_timers[handle]->_start));	
#else
    struct timeval s;
    gettimeofday(&s, 0);
    _timers[handle]->_start = (long long)s.tv_sec * (long long)1.0E3 + (long long)s.tv_usec / (long long)1.0E3;
#endif

    return SDK_SUCCESS;
}

int CTimer::stopTimer(int handle)
{
    long long n=0;

    if(handle >= (int)_timers.size())
    {
        error("Cannot reset timer. Invalid handle.");
        return SDK_FAILURE;
    }

#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER*)&(n));	
#else
    struct timeval s;
    gettimeofday(&s, 0);
    n = (long long)s.tv_sec * (long long)1.0E3+ (long long)s.tv_usec / (long long)1.0E3;
#endif

	SetThreadAffinityMask(::GetCurrentThread(), oldmask);

    n -= _timers[handle]->_start;
    _timers[handle]->_start = 0;
    _timers[handle]->_clocks += n;

    return SDK_SUCCESS;
}

double CTimer::readTimer(int handle)
{
    if(handle >= (int)_timers.size())
    {
        error("Cannot read timer. Invalid handle.");
        return SDK_FAILURE;
    }

    double reading = double(_timers[handle]->_clocks);
    reading = double(reading / _timers[handle]->_freq);

    return reading;
}


void 
	CTimer::error(const char* errorMsg)
{
	std::cout<<"Error: "<<errorMsg<<std::endl;
}

void 
	CTimer::error(std::string errorMsg)
{
	std::cout<<"Error: "<<errorMsg<<std::endl;
}

void CTimer::printfTimer( std::ostringstream &oss )
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
		if ( nItem >= 100 )
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
	oss << std::endl;
}

void CTimer::insertTimer( std::string timeString, double timeValue)
{
	if ( _timeValueList.size()>200 )
	{
		return;
	}
	_timeValueList.insert( std::make_pair(timeString, timeValue) );
}

void CTimer::warmup()
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
