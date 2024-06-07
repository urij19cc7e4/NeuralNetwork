#pragma once

#include "data.h"

namespace arithmetic
{
	template <typename T, bool initialize>
	vec<T, initialize> operator*(const mtx<T, initialize>& _mtx, const vec<T, initialize>& _vec);

	template <typename T, bool initialize>
	vec<T, initialize> operator*(const vec<T, initialize>& _vec, const mtx<T, initialize>& _mtx);
}

template <typename T, bool initialize>
class vec : public data<T, initialize>
{
protected:
	vec(T* data, uint64_t size) : data<T, initialize>()
	{
		this->_data = data;
		this->_size = size;
	}

	virtual void assign(const data<T, initialize>& o) noexcept
	{
		data<T, initialize>::assign(o);
	}

	virtual bool equal(const data<T, initialize>& o) const noexcept
	{
		return data<T, initialize>::equal(o);
	}

	virtual void zero() noexcept
	{
		data<T, initialize>::zero();
	}

	template <typename T, bool initialize>
	friend vec<T, initialize> arithmetic::operator*(const mtx<T, initialize>& _mtx, const vec<T, initialize>& _vec);

	template <typename T, bool initialize>
	friend vec<T, initialize> arithmetic::operator*(const vec<T, initialize>& _vec, const mtx<T, initialize>& _mtx);

	friend class data<T, initialize>;

public:
	vec() noexcept : data<T, initialize>() {}

	vec(uint64_t size) : data<T, initialize>(size)
	{
		if (this->_size == (uint64_t)0)
			zero();
	}

	vec(std::initializer_list<T> list) : data<T, initialize>(list.size())
	{
		if (this->_size == (uint64_t)0)
			zero();
		else
		{
			const T* elems = list.begin();
			for (uint64_t i = (uint64_t)0; i < this->_size; ++i)
				this->_data[i] = elems[i];
		}
	}

	vec(const vec& o) : data<T, initialize>(o)
	{
		if (o._data == nullptr)
			zero();
	}

	vec(vec&& o) noexcept : data<T, initialize>(std::move(o))
	{
		o.zero();
	}

	virtual ~vec() {}

	vec& operator=(const vec& o)
	{
		return (vec&)data<T, initialize>::operator=(o);
	}

	vec& operator=(vec&& o) noexcept
	{
		return (vec&)data<T, initialize>::operator=(std::move(o));
	}

	vec& operator+=(const vec& o)
	{
		return (vec&)data<T, initialize>::operator+=(o);
	}

	vec& operator-=(const vec& o)
	{
		return (vec&)data<T, initialize>::operator-=(o);
	}

	vec& operator*=(const vec& o)
	{
		return (vec&)data<T, initialize>::operator*=(o);
	}

	vec& operator/=(const vec& o)
	{
		return (vec&)data<T, initialize>::operator/=(o);
	}

	vec& operator+=(const T& sub_o)
	{
		return (vec&)data<T, initialize>::operator+=(sub_o);
	}

	vec& operator-=(const T& sub_o)
	{
		return (vec&)data<T, initialize>::operator-=(sub_o);
	}

	vec& operator*=(const T& sub_o)
	{
		return (vec&)data<T, initialize>::operator*=(sub_o);
	}

	vec& operator/=(const T& sub_o)
	{
		return (vec&)data<T, initialize>::operator/=(sub_o);
	}

	using data<T, initialize>::operator();
	using data<T, initialize>::operator[];
};