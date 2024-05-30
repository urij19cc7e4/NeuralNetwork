#pragma once

#include <cstdint>
#include <functional>
#include <tuple>

#include "data.h"

using namespace std;

template <typename T, bool initialize>
class mtx;

template <typename T, bool initialize>
class vec;

namespace arithmetic
{
	template <typename T>
	vec<T> apply_multiply(const mtx<T>& _mtx, const vec<T>& _vec, function<T(T)> apply);

	template <typename T>
	vec<T> apply_multiply(const vec<T>& _vec, const mtx<T>& _mtx, function<T(T)> apply);

	template <typename T>
	tuple<vec<T>, vec<T>> apply_multiply(const mtx<T>& _mtx, const vec<T>& _vec,
		function<T(T)> apply, function<T(T, T)> again);

	template <typename T>
	tuple<vec<T>, vec<T>> apply_multiply(const vec<T>& _vec, const mtx<T>& _mtx,
		function<T(T)> apply, function<T(T, T)> again);

	template <typename T>
	vec<T> operator*(const mtx<T>& _mtx, const vec<T>& _vec);

	template <typename T>
	vec<T> operator*(const vec<T>& _vec, const mtx<T>& _mtx);
}

template <typename T, bool initialize>
class mtx
{
private:
	T* _data;
	uint64_t _size_1;
	uint64_t _size_2;

	template <typename T>
	friend vec<T> arithmetic::apply_multiply(const mtx<T>& _mtx, const vec<T>& _vec, function<T(T)> apply);

	template <typename T>
	friend vec<T> arithmetic::apply_multiply(const vec<T>& _vec, const mtx<T>& _mtx, function<T(T)> apply);

	template <typename T>
	friend tuple<vec<T>, vec<T>> arithmetic::apply_multiply(const mtx<T>& _mtx, const vec<T>& _vec,
		function<T(T)> apply, function<T(T, T)> again);

	template <typename T>
	friend tuple<vec<T>, vec<T>> arithmetic::apply_multiply(const vec<T>& _vec, const mtx<T>& _mtx,
		function<T(T)> apply, function<T(T, T)> again);

	template <typename T>
	friend vec<T> arithmetic::operator*(const mtx<T>& _mtx, const vec<T>& _vec);

	template <typename T>
	friend vec<T> arithmetic::operator*(const vec<T>& _vec, const mtx<T>& _mtx);

public:
	mtx() noexcept : _data(nullptr), _size_1((uint64_t)0), _size_2((uint64_t)0) {}

	mtx(uint64_t size_1, uint64_t size_2) : _size_1(size_1), _size_2(size_2)
	{
		if (_size_1 == (uint64_t)0 || _size_2 == (uint64_t)0)
		{
			_data = nullptr;
			_size_1 = (uint64_t)0;
			_size_2 = (uint64_t)0;
		}
		else
			_data = initialize ? new T[_size_1 * _size_2]() : new T[_size_1 * _size_2];
	}

	mtx(initializer_list<initializer_list<T>> list) : _size_1(list.size()), _size_2(list.begin()->size())
	{
		if (_size_1 == (uint64_t)0 || _size_2 == (uint64_t)0)
		{
			_data = nullptr;
			_size_1 = (uint64_t)0;
			_size_2 = (uint64_t)0;
		}
		else
		{
			_data = initialize ? new T[_size_1 * _size_2]() : new T[_size_1 * _size_2];

			const initializer_list<T>* lists = list.begin();
			uint64_t k = (uint64_t)0;

			for (uint64_t i = (uint64_t)0; i < _size_1; ++i)
			{
				if (lists[i].size() == _size_2)
				{
					const T* elems = lists[i].begin();

					for (uint64_t j = (uint64_t)0; j < _size_2; ++j, ++k)
						_data[k] = elems[j];
				}
				else
				{
					if (_data != nullptr)
						delete[] _data;

					_data = nullptr;
					_size_1 = (uint64_t)0;
					_size_2 = (uint64_t)0;
				}
			}
		}
	}

	mtx(const mtx& o) : _size_1(o._size_1), _size_2(o._size_2)
	{
		if (o.is_empty())
		{
			_data = nullptr;
			_size_1 = (uint64_t)0;
			_size_2 = (uint64_t)0;
		}
		else
		{
			_data = initialize ? new T[_size_1 * _size_2]() : new T[_size_1 * _size_2];

			for (uint64_t i = (uint64_t)0; i < _size_1 * _size_2; ++i)
				_data[i] = o._data[i];
		}
	}

	mtx(mtx&& o) noexcept : _data(o._data), _size_1(o._size_1), _size_2(o._size_2)
	{
		o._data = nullptr;
		o._size_1 = (uint64_t)0;
		o._size_2 = (uint64_t)0;
	}

	~mtx()
	{
		if (_data != nullptr)
			delete[] _data;
	}

	uint64_t get_size_1() const noexcept
	{
		return _size_1;
	}

	uint64_t get_size_2() const noexcept
	{
		return _size_2;
	}

	bool is_empty() const noexcept
	{
		return _data == nullptr || _size_1 == (uint64_t)0 || _size_2 == (uint64_t)0;
	}

	mtx<T>& operator=(const mtx<T>& o)
	{
		if (_data != nullptr)
			delete[] _data;

		if (o.is_empty())
		{
			_data = nullptr;
			_size_1 = (uint64_t)0;
			_size_2 = (uint64_t)0;
		}
		else
		{
			_data = initialize ? new T[o._size_1 * o._size_2]() : new T[o._size_1 * o._size_2];
			_size_1 = o._size_1;
			_size_2 = o._size_2;

			for (uint64_t i = (uint64_t)0; i < _size_1 * _size_2; ++i)
				_data[i] = o._data[i];
		}

		return *this;
	}

	mtx<T>& operator=(mtx<T>&& o) noexcept
	{
		if (_data != nullptr)
			delete[] _data;

		_data = o._data;
		_size_1 = o._size_1;
		_size_2 = o._size_2;

		o._data = nullptr;
		o._size_1 = (uint64_t)0;
		o._size_2 = (uint64_t)0;

		return *this;
	}

	const T& operator()(uint64_t index_1, uint64_t index_2) const
	{
		return _data[_size_2 * index_1 + index_2];
	}

	T& operator()(uint64_t index_1, uint64_t index_2)
	{
		return _data[_size_2 * index_1 + index_2];
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