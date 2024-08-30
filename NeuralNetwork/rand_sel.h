#pragma once

#include <exception>
#include <list>
#include <random>

template <typename T>
class rand_sel_i
{
public:
	virtual T next() = 0;

	virtual void reset() = 0;
};

template <typename T, bool strict>
class rand_sel;

template <typename T>
class rand_sel<typename T, false> : public rand_sel_i<T>
{
private:
	std::mt19937_64 _rand_gen;
	std::uniform_int_distribution<T> _dstr;

public:
	rand_sel() = delete;

	rand_sel(T max, T min) : _rand_gen((std::random_device())()), _dstr(min, max) {}

	rand_sel(const rand_sel& o) = delete;

	rand_sel(rand_sel&& o) = delete;

	~rand_sel() {}

	virtual T next() override
	{
		return _dstr(_rand_gen);
	}

	virtual void reset() override {}
};

template <typename T>
class rand_sel<typename T, true> : public rand_sel_i<T>
{
private:
	std::mt19937_64 _rand_gen;
	std::list<T> _rest;
	std::list<T> _used;

public:
	rand_sel() = delete;

	rand_sel(T max, T min) : _rand_gen((std::random_device())())
	{
		for (T i = min;; ++i)
		{
			_rest.push_back(i);

			if (i == max)
				break;
		}
	}

	rand_sel(const rand_sel& o) = delete;

	rand_sel(rand_sel&& o) = delete;

	~rand_sel() {}

	virtual T next() override
	{
		if (_rest.empty())
			throw std::exception("No elements left");
		else
		{
			std::uniform_int_distribution<T> dstr((uint64_t)0, _rest.size() - (uint64_t)1);

			std::list<T>::template iterator iter = _rest.begin();
			std::advance(iter, dstr(_rand_gen));

			T result = *iter;
			_used.splice(_used.end(), _rest, iter);

			return result;
		}
	}

	virtual void reset() override
	{
		_rest.splice(_rest.end(), _used);
	}
};