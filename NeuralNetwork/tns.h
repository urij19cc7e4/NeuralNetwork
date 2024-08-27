#pragma once

#include "data.h"

template <typename T, bool initialize>
class tns : public data<T, initialize>
{
protected:
	uint64_t _size_1;
	uint64_t _size_2;
	uint64_t _size_3;

	tns(T* data, uint64_t size_1, uint64_t size_2, uint64_t size_3)
		: data<T, initialize>(), _size_1(size_1), _size_2(size_2), _size_3(size_3)
	{
		this->_data = data;
		this->_size = _size_1 * _size_2 * _size_3;
	}

	virtual void assign(const data<T, initialize>& o) noexcept
	{
		const tns& oo = (const tns&)o;
		data<T, initialize>::assign(o);

		_size_1 = oo._size_1;
		_size_2 = oo._size_2;
		_size_3 = oo._size_3;
	}

	virtual bool equal(const data<T, initialize>& o) const noexcept
	{
		const tns& oo = (const tns&)o;
		return data<T, initialize>::equal(o) && _size_1 == oo._size_1 && _size_2 == oo._size_2 && _size_3 == oo._size_3;
	}

	virtual void zero() noexcept
	{
		data<T, initialize>::zero();

		_size_1 = (uint64_t)0;
		_size_2 = (uint64_t)0;
		_size_3 = (uint64_t)0;
	}

	friend class data<T, initialize>;

public:
	tns() noexcept : data<T, initialize>(), _size_1((uint64_t)0), _size_2((uint64_t)0), _size_3((uint64_t)0) {}

	tns(uint64_t size_1, uint64_t size_2, uint64_t size_3)
		: data<T, initialize>(size_1 * size_2 * size_3), _size_1(size_1), _size_2(size_2), _size_3(size_3)
	{
		if (this->_size == (uint64_t)0)
			zero();
	}

	tns(std::initializer_list<std::initializer_list<std::initializer_list<T>>> list)
		: data<T, initialize>(list.size() * list.begin()->size() * list.begin()->begin()->size()),
		_size_1(list.size()), _size_2(list.begin()->size()), _size_3(list.begin()->begin()->size())
	{
		if (this->_size == (uint64_t)0)
			zero();
		else
		{
			const std::initializer_list<std::initializer_list<T>>* lists_1 = list.begin();

			for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < _size_1; ++i)
				if (lists_1[i].size() == _size_2)
				{
					const std::initializer_list<T>* lists_2 = lists_1[i].begin();

					for (uint64_t j = (uint64_t)0; j < _size_2; ++j)
						if (lists_2[j].size() == _size_3)
						{
							const T* elems = lists_2[j].begin();

							for (uint64_t k = (uint64_t)0; k < _size_3; ++k, ++l)
								this->_data[l] = elems[k];
						}
						else
						{
							delete[] this->_data;
							zero();

							throw std::exception("3-dims array required, jagged arrays are not supported");
						}
				}
				else
				{
					delete[] this->_data;
					zero();

					throw std::exception("3-dims array required, jagged arrays are not supported");
				}
		}
	}

	tns(const tns& o) : data<T, initialize>(o), _size_1(o._size_1), _size_2(o._size_2), _size_3(o._size_3)
	{
		if (o._data == nullptr)
			zero();
	}

	tns(tns&& o) noexcept : data<T, initialize>(std::move(o)), _size_1(o._size_1), _size_2(o._size_2), _size_3(o._size_3)
	{
		o.zero();
	}

	virtual ~tns() {}

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

	tns& operator=(const tns& o)
	{
		return (tns&)data<T, initialize>::operator=(o);
	}

	tns& operator=(tns&& o) noexcept
	{
		return (tns&)data<T, initialize>::operator=(std::move(o));
	}

	tns& operator=(const T& sub_o)
	{
		return (tns&)data<T, initialize>::operator=(sub_o);
	}

	tns& operator+=(const tns& o)
	{
		return (tns&)data<T, initialize>::operator+=(o);
	}

	tns& operator-=(const tns& o)
	{
		return (tns&)data<T, initialize>::operator-=(o);
	}

	tns& operator*=(const tns& o)
	{
		return (tns&)data<T, initialize>::operator*=(o);
	}

	tns& operator/=(const tns& o)
	{
		return (tns&)data<T, initialize>::operator/=(o);
	}

	tns& operator+=(const T& sub_o)
	{
		return (tns&)data<T, initialize>::operator+=(sub_o);
	}

	tns& operator-=(const T& sub_o)
	{
		return (tns&)data<T, initialize>::operator-=(sub_o);
	}

	tns& operator*=(const T& sub_o)
	{
		return (tns&)data<T, initialize>::operator*=(sub_o);
	}

	tns& operator/=(const T& sub_o)
	{
		return (tns&)data<T, initialize>::operator/=(sub_o);
	}

	const T& operator()(uint64_t index_1, uint64_t index_2, uint64_t index_3) const
	{
		return this->_data[_size_3 * (_size_2 * index_1 + index_2) + index_3];
	}

	T& operator()(uint64_t index_1, uint64_t index_2, uint64_t index_3)
	{
		return this->_data[_size_3 * (_size_2 * index_1 + index_2) + index_3];
	}

	using data<T, initialize>::operator();
	using data<T, initialize>::operator[];
};