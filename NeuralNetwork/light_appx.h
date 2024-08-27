#pragma once

#include <cmath>
#include <cstdint>

class light_appx
{
private:
	uint64_t _count;
	uint64_t _pos;

	double _ln_start;
	double _ln_end;
	double _start;
	double _end;

	double calculate() const noexcept
	{
		if (_start == _end || _count == (uint64_t)0 || _count == (uint64_t)1)
			return (_start + _end) / (double)2;
		else if (_ln_start == _ln_end)
			return (_end - _start) * (double)_pos / (double)(_count - (uint64_t)1) + _start;
		else
			return (_ln_end - log((exp(_ln_end) - exp(_ln_start))
				* (double)_pos / (double)(_count - (uint64_t)1) + exp(_ln_start)))
			* (_end - _start) / (_ln_start - _ln_end) + _end;
	}

public:
	light_appx() noexcept
		: _count((uint64_t)0), _pos((uint64_t)0), _ln_start((double)0),
		_ln_end((double)0), _start((double)0), _end((double)0) {}

	light_appx(double start, double end, uint64_t count = (uint64_t)100,
		double ln_start = (double)0, double ln_end = (double)3) noexcept
		: _count(count), _pos((uint64_t)0), _ln_start(ln_start),
		_ln_end(ln_end), _start(start), _end(end) {}

	light_appx(const light_appx& o) noexcept
		: _count(o._count), _pos(o._pos), _ln_start(o._ln_start),
		_ln_end(o._ln_end), _start(o._start), _end(o._end) {}

	light_appx(light_appx&& o) noexcept
		: _count(o._count), _pos(o._pos), _ln_start(o._ln_start),
		_ln_end(o._ln_end), _start(o._start), _end(o._end) {}

	~light_appx() noexcept {}

	double current() const noexcept
	{
		return calculate();
	}

	double forward() noexcept
	{
		double result = calculate();

		if (_count == (uint64_t)0)
			_pos = (uint64_t)0;
		else if (_pos == _count - (uint64_t)1)
			_pos = (uint64_t)0;
		else
			++_pos;

		return result;
	}

	double backward() noexcept
	{
		double result = calculate();

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