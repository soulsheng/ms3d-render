#pragma once

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>


	typedef std::multimap<std::string, double> TimerValueList;
	typedef std::multimap<std::string, double>::iterator TimerValueListItr;

class CTimerOMP
{
public:
	int createTimer();
	int resetTimer(int handle=0);
	int startTimer(int handle=0);
	int stopTimer(int handle=0);
	double readTimer(int handle=0);


	void printfTimer(  std::ostringstream &oss );
	void insertTimer(std::string timeString, double timeValue);
	void clear();
protected:
	void error(const char* errorMsg);
	void error(std::string errorMsg);
	void warmup(); // before start 
private:
	double _start;	/**< start point ticks*/
	double _end;	/**< _clocks number of ticks at end*/

	TimerValueList	_timeValueList;
	DWORD_PTR oldmask;

};


