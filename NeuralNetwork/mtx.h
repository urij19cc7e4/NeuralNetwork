#pragma once

#include "data.h"

template <typename T, bool initialize>
class mtx : public data<T, initialize>
{
protected:
	uint64_t _size_1;
	uint64_t _size_2;

	mtx(T* data, uint64_t size_1, uint64_t size_2) : data<T, initialize>(), _size_1(size_1), _size_2(size_2)
	{
		this->_data = data;
		this->_size = _size_1 * _size_2;
	}

	virtual void assign(const data<T, initialize>& o) noexcept
	{
		const mtx& oo = (const mtx&)o;
		data<T, initialize>::assign(o);

		_size_1 = oo._size_1;
		_size_2 = oo._size_2;
	}

	virtual bool equal(const data<T, initialize>& o) const noexcept
	{
		const mtx& oo = (const mtx&)o;
		return data<T, initialize>::equal(o) && _size_1 == oo._size_1 && _size_2 == oo._size_2;
	}

	virtual void zero() noexcept
	{
		data<T, initialize>::zero();

		_size_1 = (uint64_t)0;
		_size_2 = (uint64_t)0;
	}

	friend class data<T, initialize>;

public:
	mtx() noexcept : data<T, initialize>(), _size_1((uint64_t)0), _size_2((uint64_t)0) {}

	mtx(uint64_t size_1, uint64_t size_2) : data<T, initialize>(size_1 * size_2), _size_1(size_1), _size_2(size_2)
	{
		if (this->_size == (uint64_t)0)
			zero();
	}

	mtx(std::initializer_list<std::initializer_list<T>> list)
		: data<T, initialize>(list.size() * list.begin()->size()), _size_1(list.size()), _size_2(list.begin()->size())
	{
		if (this->_size == (uint64_t)0)
			zero();
		else
		{
			const std::initializer_list<T>* lists = list.begin();

			for (uint64_t i = (uint64_t)0, k = (uint64_t)0; i < _size_1; ++i)
				if (lists[i].size() == _size_2)
				{
					const T* elems = lists[i].begin();

					for (uint64_t j = (uint64_t)0; j < _size_2; ++j, ++k)
						this->_data[k] = elems[j];
				}
				else
				{
					delete[] this->_data;
					zero();

					throw std::exception("2-dims array required, jagged arrays are not supported");
				}
		}
	}

	mtx(const mtx& o) : data<T, initialize>(o), _size_1(o._size_1), _size_2(o._size_2)
	{
		if (o._data == nullptr)
			zero();
	}

	mtx(mtx&& o) noexcept : data<T, initialize>(std::move(o)), _size_1(o._size_1), _size_2(o._size_2)
	{
		o.zero();
	}

	virtual ~mtx() {}

	uint64_t get_size_1() const noexcept
	{
		return _size_1;
	}

	uint64_t get_size_2() const noexcept
	{
		return _size_2;
	}

	mtx& operator=(const mtx& o)
	{
		return (mtx&)data<T, initialize>::operator=(o);
	}

	mtx& operator=(mtx&& o) noexcept
	{
		return (mtx&)data<T, initialize>::operator=(std::move(o));
	}

	mtx& operator=(const T& sub_o)
	{
		return (mtx&)data<T, initialize>::operator=(sub_o);
	}

	mtx& operator+=(const mtx& o)
	{
		return (mtx&)data<T, initialize>::operator+=(o);
	}

	mtx& operator-=(const mtx& o)
	{
		return (mtx&)data<T, initialize>::operator-=(o);
	}

	mtx& operator*=(const mtx& o)
	{
		return (mtx&)data<T, initialize>::operator*=(o);
	}

	mtx& operator/=(const mtx& o)
	{
		return (mtx&)data<T, initialize>::operator/=(o);
	}

	mtx& operator+=(const T& sub_o)
	{
		return (mtx&)data<T, initialize>::operator+=(sub_o);
	}

	mtx& operator-=(const T& sub_o)
	{
		return (mtx&)data<T, initialize>::operator-=(sub_o);
	}

	mtx& operator*=(const T& sub_o)
	{
		return (mtx&)data<T, initialize>::operator*=(sub_o);
	}

	mtx& operator/=(const T& sub_o)
	{
		return (mtx&)data<T, initialize>::operator/=(sub_o);
	}

	const T& operator()(uint64_t index_1, uint64_t index_2) const
	{
		return this->_data[_size_2 * index_1 + index_2];
	}

	T& operator()(uint64_t index_1, uint64_t index_2)
	{
		return this->_data[_size_2 * index_1 + index_2];
	}

	using data<T, initialize>::operator();
	using data<T, initialize>::operator[];
};