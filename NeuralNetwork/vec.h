#pragma once

#include "data.h"

template <typename T, bool initialize>
class mtx;

template <typename T, bool initialize>
class vec;

namespace arithmetic
{
	template <typename T, bool initialize>
	vec<T, initialize> operator*(const mtx<T, initialize>& _mtx, const vec<T, initialize>& _vec);

	template <typename T, bool initialize>
	vec<T, initialize> operator*(const vec<T, initialize>& _vec, const mtx<T, initialize>& _mtx);
}

template <typename T, bool initialize>
class vec
{
private:
	T* _data;
	uint64_t _size;

	template <typename T, bool initialize>
	friend vec<T, initialize> arithmetic::operator*(const mtx<T, initialize>& _mtx, const vec<T, initialize>& _vec);

	template <typename T, bool initialize>
	friend vec<T, initialize> arithmetic::operator*(const vec<T, initialize>& _vec, const mtx<T, initialize>& _mtx);

public:
	vec() noexcept : _data(nullptr), _size((uint64_t)0) {}

	vec(uint64_t size) : _size(size)
	{
		_data = size == (uint64_t)0 ? nullptr : initialize ? new T[_size]() : new T[_size];
	}

	vec(std::initializer_list<T> list) : _size(list.size())
	{
		if (_size == (uint64_t)0)
			_data = nullptr;
		else
		{
			_data = initialize ? new T[_size]() : new T[_size];

			const T* elems = list.begin();
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] = elems[i];
		}
	}

	vec(const vec& o) : _size(o._size)
	{
		if (o.is_empty())
		{
			_data = nullptr;
			_size = (uint64_t)0;
		}
		else
		{
			_data = initialize ? new T[_size]() : new T[_size];

			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] = o._data[i];
		}
	}

	vec(vec&& o) noexcept : _data(o._data), _size(o._size)
	{
		o._data = nullptr;
		o._size = (uint64_t)0;
	}

	~vec()
	{
		if (_data != nullptr)
			delete[] _data;
	}

	uint64_t get_size() const noexcept
	{
		return _size;
	}

	bool is_empty() const noexcept
	{
		return _data == nullptr || _size == (uint64_t)0;
	}

	vec& operator=(const vec& o)
	{
		if (_data != nullptr)
			delete[] _data;

		if (o.is_empty())
		{
			_data = nullptr;
			_size = (uint64_t)0;
		}
		else
		{
			_data = initialize ? new T[o._size]() : new T[o._size];
			_size = o._size;

			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] = o._data[i];
		}

		return *this;
	}

	vec& operator=(vec&& o) noexcept
	{
		if (_data != nullptr)
			delete[] _data;

		_data = o._data;
		_size = o._size;

		o._data = nullptr;
		o._size = (uint64_t)0;

		return *this;
	}

	vec& operator+=(const vec& o)
	{
		if (_size == o._size)
		{
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] += o._data[i];

			return *this;
		}
		else
			throw std::exception(vec_sizes_error);
	}

	vec& operator-=(const vec& o)
	{
		if (_size == o._size)
		{
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] -= o._data[i];

			return *this;
		}
		else
			throw std::exception(vec_sizes_error);
	}

	const T& operator()(uint64_t index) const
	{
		return _data[index];
	}

	T& operator()(uint64_t index)
	{
		return _data[index];
	}
};