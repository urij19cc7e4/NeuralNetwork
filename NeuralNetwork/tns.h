#pragma once

#include "data.h"

template <typename T, bool initialize>
class tns;

template <typename T, bool initialize>
class mtx;

template <typename T, bool initialize>
class vec;

template <typename T, bool initialize>
class tns
{
private:
	T* _data;
	uint64_t _size;
	uint64_t _size_1;
	uint64_t _size_2;
	uint64_t _size_3;

public:
	tns() noexcept
		: _data(nullptr), _size((uint64_t)0), _size_1((uint64_t)0), _size_2((uint64_t)0), _size_3((uint64_t)0) {}

	tns(uint64_t size_1, uint64_t size_2, uint64_t size_3)
		: _size(size_1* size_2* size_3), _size_1(size_1), _size_2(size_2), _size_3(size_3)
	{
		if (_size == (uint64_t)0)
		{
			_data = nullptr;
			_size_1 = (uint64_t)0;
			_size_2 = (uint64_t)0;
			_size_3 = (uint64_t)0;
		}
		else
			_data = initialize ? new T[_size]() : new T[_size];
	}

	tns(std::initializer_list<std::initializer_list<std::initializer_list<T>>> list)
		: _size_1(list.size()), _size_2(list.begin()->size()), _size_3(list.begin()->begin()->size())
	{
		_size = _size_1 * _size_2 * _size_3;

		if (_size == (uint64_t)0)
		{
			_data = nullptr;
			_size_1 = (uint64_t)0;
			_size_2 = (uint64_t)0;
			_size_3 = (uint64_t)0;
		}
		else
		{
			_data = initialize ? new T[_size]() : new T[_size];

			const std::initializer_list<std::initializer_list<T>>* lists_1 = list.begin();
			uint64_t l = (uint64_t)0;

			for (uint64_t i = (uint64_t)0; i < _size_1; ++i)
				if (lists_1[i].size() == _size_2)
				{
					const std::initializer_list<T>* lists_2 = lists_1[i].begin();

					for (uint64_t j = (uint64_t)0; j < _size_2; ++j)
						if (lists_2[j].size() == _size_3)
						{
							const T* elems = lists_2[j].begin();

							for (uint64_t k = (uint64_t)0; k < _size_3; ++k, ++l)
								_data[l] = elems[k];
						}
						else
						{
							if (_data != nullptr)
								delete[] _data;

							_data = nullptr;
							_size = (uint64_t)0;
							_size_1 = (uint64_t)0;
							_size_2 = (uint64_t)0;
							_size_3 = (uint64_t)0;

							return;
						}
				}
				else
				{
					if (_data != nullptr)
						delete[] _data;

					_data = nullptr;
					_size = (uint64_t)0;
					_size_1 = (uint64_t)0;
					_size_2 = (uint64_t)0;
					_size_3 = (uint64_t)0;

					return;
				}
		}
	}

	tns(const tns& o) : _size(o._size), _size_1(o._size_1), _size_2(o._size_2), _size_3(o._size_3)
	{
		if (o._data == nullptr)
		{
			_data = nullptr;
			_size = (uint64_t)0;
			_size_1 = (uint64_t)0;
			_size_2 = (uint64_t)0;
			_size_3 = (uint64_t)0;
		}
		else
		{
			_data = initialize ? new T[_size]() : new T[_size];

			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] = o._data[i];
		}
	}

	tns(tns&& o) noexcept : _data(o._data), _size(o._size), _size_1(o._size_1), _size_2(o._size_2), _size_3(o._size_3)
	{
		o._data = nullptr;
		o._size = (uint64_t)0;
		o._size_1 = (uint64_t)0;
		o._size_2 = (uint64_t)0;
		o._size_3 = (uint64_t)0;
	}

	~tns()
	{
		if (_data != nullptr)
			delete[] _data;
	}

	uint64_t get_size() const noexcept
	{
		return _size;
	}

	uint64_t get_size_1() const noexcept
	{
		return _size_1;
	}

	uint64_t get_size_2() const noexcept
	{
		return _size_2;
	}

	uint64_t get_size_3() const noexcept
	{
		return _size_3;
	}

	bool is_empty() const noexcept
	{
		return _data == nullptr;
	}

	tns& operator=(const tns& o)
	{
		if (_data != nullptr)
			delete[] _data;

		if (o._data == nullptr)
		{
			_data = nullptr;
			_size = (uint64_t)0;
			_size_1 = (uint64_t)0;
			_size_2 = (uint64_t)0;
			_size_3 = (uint64_t)0;
		}
		else
		{
			_data = initialize ? new T[o._size]() : new T[o._size];
			_size = o._size;
			_size_1 = o._size_1;
			_size_2 = o._size_2;
			_size_3 = o._size_3;

			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] = o._data[i];
		}

		return *this;
	}

	tns& operator=(tns&& o) noexcept
	{
		if (_data != nullptr)
			delete[] _data;

		_data = o._data;
		_size = o._size;
		_size_1 = o._size_1;
		_size_2 = o._size_2;
		_size_3 = o._size_3;

		o._data = nullptr;
		o._size = (uint64_t)0;
		o._size_1 = (uint64_t)0;
		o._size_2 = (uint64_t)0;
		o._size_3 = (uint64_t)0;

		return *this;
	}

	tns& operator+=(const tns& o)
	{
		if (_data == nullptr && o._data == nullptr)
			return *this;
		else if (_size == o._size && _size_1 == o._size_1 && _size_2 == o._size_2 && _size_3 == o._size_3)
		{
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] += o._data[i];

			return *this;
		}
		else
			throw std::exception(tns_sizes_error);
	}

	tns& operator-=(const tns& o)
	{
		if (_data == nullptr && o._data == nullptr)
			return *this;
		else if (_size == o._size && _size_1 == o._size_1 && _size_2 == o._size_2 && _size_3 == o._size_3)
		{
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] -= o._data[i];

			return *this;
		}
		else
			throw std::exception(tns_sizes_error);
	}

	tns& operator+=(const T& sub_o)
	{
		if (_data != nullptr)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] += sub_o;

		return *this;
	}

	tns& operator-=(const T& sub_o)
	{
		if (_data != nullptr)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] -= sub_o;

		return *this;
	}

	tns& operator*=(const T& sub_o)
	{
		if (_data != nullptr)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] *= sub_o;

		return *this;
	}

	tns& operator/=(const T& sub_o)
	{
		if (_data != nullptr)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] /= sub_o;

		return *this;
	}

	const T& operator()(uint64_t index_1, uint64_t index_2, uint64_t index_3) const
	{
		return _data[_size_3 * (_size_2 * index_1 + index_2) + index_3];
	}

	T& operator()(uint64_t index_1, uint64_t index_2, uint64_t index_3)
	{
		return _data[_size_3 * (_size_2 * index_1 + index_2) + index_3];
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