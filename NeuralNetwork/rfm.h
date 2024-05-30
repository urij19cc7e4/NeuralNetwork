#pragma once

#include <cstdint>
#include <exception>
#include <functional>
#include <tuple>

#include "mtx.h"
#include "vec.h"

using namespace std;

namespace arithmetic
{
	constexpr char mtx_sizes_error[] = "Incorrect sizes of matrices";

	template <typename T>
	vec<T> apply_multiply(const mtx<T>& _mtx, const vec<T>& _vec, function<T(T)> apply)
	{
		if (_vec._size == _mtx._size_2)
		{
			uint64_t size_1 = _mtx._size_1, size_2 = _mtx._size_2;
			vec<T> result(size_1);

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
			{
				T cell = (T)0;

				for (uint64_t j = (uint64_t)0, k = i * size_2; j < size_2; ++j, ++k)
					cell += _vec._data[j] * _mtx._data[k];

				result._data[i] = apply(cell);
			}

			return result;
		}
		else
			throw exception(mtx_sizes_error);
	}

	template <typename T>
	vec<T> apply_multiply(const vec<T>& _vec, const mtx<T>& _mtx, function<T(T)> apply)
	{
		if (_vec._size == _mtx._size_1)
		{
			uint64_t size_1 = _mtx._size_1, size_2 = _mtx._size_2;
			vec<T> result(size_2);

			for (uint64_t i = (uint64_t)0; i < size_2; ++i)
			{
				T cell = (T)0;

				for (uint64_t j = (uint64_t)0, k = i; j < size_1; ++j, k += size_2)
					cell += _vec._data[j] * _mtx._data[k];

				result._data[i] = apply(cell);
			}

			return result;
		}
		else
			throw exception(mtx_sizes_error);
	}

	template <typename T>
	tuple<vec<T>, vec<T>> apply_multiply(const mtx<T>& _mtx, const vec<T>& _vec,
		function<T(T)> apply, function<T(T, T)> again)
	{
		if (_vec._size == _mtx._size_2)
		{
			uint64_t size_1 = _mtx._size_1, size_2 = _mtx._size_2;
			tuple<vec<T>, vec<T>> result{ vec<T>(size_1), vec<T>(size_1) };

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
			{
				T cell = (T)0;

				for (uint64_t j = (uint64_t)0, k = i * size_2; j < size_2; ++j, ++k)
					cell += _vec._data[j] * _mtx._data[k];

				T __cell = apply(cell);
				get<(size_t)0>(result)._data[i] = __cell;
				get<(size_t)1>(result)._data[i] = again(cell, __cell);
			}

			return result;
		}
		else
			throw exception(mtx_sizes_error);
	}

	template <typename T>
	tuple<vec<T>, vec<T>> apply_multiply(const vec<T>& _vec, const mtx<T>& _mtx,
		function<T(T)> apply, function<T(T, T)> again)
	{
		if (_vec._size == _mtx._size_1)
		{
			uint64_t size_1 = _mtx._size_1, size_2 = _mtx._size_2;
			tuple<vec<T>, vec<T>> result{ vec<T>(size_2), vec<T>(size_2) };

			for (uint64_t i = (uint64_t)0; i < size_2; ++i)
			{
				T cell = (T)0;

				for (uint64_t j = (uint64_t)0, k = i; j < size_1; ++j, k += size_2)
					cell += _vec._data[j] * _mtx._data[k];

				T __cell = apply(cell);
				get<(size_t)0>(result)._data[i] = __cell;
				get<(size_t)1>(result)._data[i] = again(cell, __cell);
			}

			return result;
		}
		else
			throw exception(mtx_sizes_error);
	}

	template <typename T>
	vec<T> operator*(const mtx<T>& _mtx, const vec<T>& _vec)
	{
		if (_vec._size == _mtx._size_2)
		{
			uint64_t size_1 = _mtx._size_1, size_2 = _mtx._size_2;
			vec<T> result(size_1);

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
			throw exception(mtx_sizes_error);
	}

	template <typename T>
	vec<T> operator*(const vec<T>& _vec, const mtx<T>& _mtx)
	{
		if (_vec._size == _mtx._size_1)
		{
			uint64_t size_1 = _mtx._size_1, size_2 = _mtx._size_2;
			vec<T> result(size_2);

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
			throw exception(mtx_sizes_error);
	}
}