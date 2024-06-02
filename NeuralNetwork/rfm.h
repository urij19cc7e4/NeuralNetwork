#pragma once

#include "data.h"
#include "mtx.h"
#include "vec.h"

namespace arithmetic
{
	template <typename T, bool initialize>
	vec<T, initialize> operator*(const mtx<T, initialize>& _mtx, const vec<T, initialize>& _vec)
	{
		if (_vec._size == _mtx._size_2)
		{
			uint64_t size_1 = _mtx._size_1, size_2 = _mtx._size_2;
			vec<T, initialize> result(size_1);

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
			{
				T cell = (T)0;

				for (uint64_t j = (uint64_t)0, k = i * size_2; j < size_2; ++j, ++k)
					cell += _vec._data[j] * _mtx._data[k];

				result._data[i] = cell;
			}

			return result;
		}
		else
			throw std::exception(mtx_vec_sizes_error);
	}

	template <typename T, bool initialize>
	vec<T, initialize> operator*(const vec<T, initialize>& _vec, const mtx<T, initialize>& _mtx)
	{
		if (_vec._size == _mtx._size_1)
		{
			uint64_t size_1 = _mtx._size_1, size_2 = _mtx._size_2;
			vec<T, initialize> result(size_2);

			for (uint64_t i = (uint64_t)0; i < size_2; ++i)
			{
				T cell = (T)0;

				for (uint64_t j = (uint64_t)0, k = i; j < size_1; ++j, k += size_2)
					cell += _vec._data[j] * _mtx._data[k];

				result._data[i] = cell;
			}

			return result;
		}
		else
			throw std::exception(mtx_vec_sizes_error);
	}
}