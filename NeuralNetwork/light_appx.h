#pragma once

#include <cmath>
#include <cstdint>

namespace light_appx
{
	template <double ln_start = (double)0, double ln_end = (double)3>
	class light_appx
	{
	private:
		uint64_t _count;
		uint64_t _pos;
		double _start;
		double _end;

		double calculate() const
		{
			if (ln_start == ln_end || _start == _end || _count == (uint64_t)0)
				return _start;
			else
			{
				double ln_major = ln_start < ln_end ? ln_end : ln_start;
				double ln_minor = ln_start < ln_end ? ln_start : ln_end;
				double major = _start < _end ? _end : _start;
				double minor = _start < _end ? _start : _end;
				double value = (ln_major - log((exp(ln_major) - exp(ln_minor))
					* (double)_pos / (double)(_count - (uint64_t)1) + exp(ln_minor)))
					* (major - minor) / (ln_major - ln_minor);

				return _start < _end ? _end - value : _end + value;
			}
		}

	public:
		light_appx() noexcept
			: _count((uint64_t)0), _pos((uint64_t)0), _start((double)0), _end((double)0) {}

		light_appx(double start, double end, uint64_t count = (uint64_t)100) noexcept
			: _count(count), _pos((uint64_t)0), _start(start), _end(end) {}

		light_appx(const light_appx& o) noexcept
			: _count(o._count), _pos(o._pos), _start(o._start), _end(o._end) {}

		light_appx(light_appx&& o) noexcept
			: _count(o._count), _pos(o._pos), _start(o._start), _end(o._end) {}

		~light_appx() noexcept {}

		double current() const
		{
			return calculate();
		}

		double forward()
		{
			double result = calculate();

			if (_pos == _count - (uint64_t)1)
				_pos = (uint64_t)0;
			else
				++_pos;

			return result;
		}

		double backward()
		{
			double result = calculate();

			if (_pos == (uint64_t)0)
				_pos = _count - (uint64_t)1;
			else
				--_pos;

			return result;
		}

		void reset()
		{
			_pos = (uint64_t)0;
		}

		light_appx& operator=(const light_appx& o) noexcept
		{
			_count = o._count;
			_pos = o._pos;
			_start = o._start;
			_end = o._end;
		}

		light_appx& operator=(light_appx&& o) noexcept
		{
			_count = o._count;
			_pos = o._pos;
			_start = o._start;
			_end = o._end;
		}
	};
}