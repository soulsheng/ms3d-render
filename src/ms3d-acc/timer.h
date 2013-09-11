#pragma once

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>
	/**
	 * Timer
	 * struct to handle time measuring functionality
	 */
	struct Timer
	{
		std::string name;   /**< name name of time object*/
		long long _freq;	/**< _freq frequency*/
		long long _clocks;	/**< _clocks number of ticks at end*/
		long long _start;	/**< _start start point ticks*/
	};

	typedef std::multimap<std::string, double> TimerValueList;
	typedef std::multimap<std::string, double>::iterator TimerValueListItr;

class CTimer
{
public:
	int createTimer();
	int resetTimer(int handle);
	int startTimer(int handle);
	int stopTimer(int handle);
	double readTimer(int handle);


	void printfTimer( std::ostringstream &oss );
	void insertTimer(std::string timeString, double timeValue);

protected:
	void error(const char* errorMsg);
	void error(std::string errorMsg);
	void warmup();

private:
	std::vector<Timer*> _timers;      /**< _timers vector to Timer objects */

	TimerValueList	_timeValueList;

	DWORD_PTR oldmask;
};


