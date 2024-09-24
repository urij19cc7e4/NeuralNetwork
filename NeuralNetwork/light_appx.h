#pragma once

#include <cmath>
#include "nn_params.h"

class light_appx
{
private:
	uint64_t _count;
	uint64_t _pos;

	FLT _ln_start;
	FLT _ln_end;
	FLT _start;
	FLT _end;

	FLT calculate() const noexcept
	{
		if (_start == _end || _count == (uint64_t)0 || _count == (uint64_t)1)
			return (_start + _end) / (FLT)2;
		else if (_ln_start == _ln_end)
			return (_end - _start) * (FLT)_pos / (FLT)(_count - (uint64_t)1) + _start;
		else
			return (_ln_end - log((exp(_ln_end) - exp(_ln_start)) * (FLT)_pos
				/ (FLT)(_count - (uint64_t)1) + exp(_ln_start)))
				* (_end - _start) / (_ln_start - _ln_end) + _end;
	}

public:
	light_appx() noexcept
		: _count((uint64_t)0), _pos((uint64_t)0), _ln_start((FLT)0), _ln_end((FLT)0), _start((FLT)0), _end((FLT)0)
	{}

	light_appx(FLT start, FLT end, uint64_t count = (uint64_t)100, FLT ln_start = (FLT)0, FLT ln_end = (FLT)3) noexcept
		: _count(count), _pos((uint64_t)0), _ln_start(ln_start), _ln_end(ln_end), _start(start), _end(end)
	{}

	light_appx(const light_appx& o) noexcept
		: _count(o._count), _pos(o._pos), _ln_start(o._ln_start), _ln_end(o._ln_end), _start(o._start), _end(o._end)
	{}

	light_appx(light_appx&& o) noexcept
		: _count(o._count), _pos(o._pos), _ln_start(o._ln_start), _ln_end(o._ln_end), _start(o._start), _end(o._end)
	{}

	~light_appx() noexcept {}

	FLT current() const noexcept
	{
		return calculate();
	}

	FLT forward() noexcept
	{
		FLT result = calculate();

		if (_count == (uint64_t)0)
			_pos = (uint64_t)0;
		else if (_pos == _count - (uint64_t)1)
			_pos = (uint64_t)0;
		else
			++_pos;

		return result;
	}

	FLT backward() noexcept
	{
		FLT result = calculate();

		if (_count == (uint64_t)0)
			_pos = (uint64_t)0;
		else if (_pos == (uint64_t)0)
			_pos = _count - (uint64_t)1;
		else
			--_pos;

		return result;
	}

	void reset() noexcept
	{
		_pos = (uint64_t)0;
	}

	light_appx& operator=(const light_appx& o) noexcept
	{
		_count = o._count;
		_pos = o._pos;
		_ln_start = o._ln_start;
		_ln_end = o._ln_end;
		_start = o._start;
		_end = o._end;

		return *this;
	}

	light_appx& operator=(light_appx&& o) noexcept
	{
		_count = o._count;
		_pos = o._pos;
		_ln_start = o._ln_start;
		_ln_end = o._ln_end;
		_start = o._start;
		_end = o._end;

		return *this;
	}
};