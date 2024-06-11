#pragma once

#include "data.h"
#include "tns.h"
#include "mtx.h"
#include "vec.h"

namespace arithmetic
{
	template <typename T, bool initialize>
	tns<T, initialize> convolute(const mtx<T, initialize>& _data, const tns<T, initialize>& _core)
	{
		if (_data.is_empty() || _core.is_empty())
			return tns<T, initialize>();
		else
		{
			tns<T, initialize> result(_data._size_1 - _core._size_2 + (uint64_t)1,
				_data._size_2 - _core._size_3 + (uint64_t)1, _core._size_1);

			__assume(result._size_3 == _core._size_1);
			for (uint64_t i = (uint64_t)0; i < result._size_3; ++i)
				for (uint64_t j = (uint64_t)0; j < result._size_1; ++j)
					for (uint64_t k = (uint64_t)0; k < result._size_2; ++k)
					{
						T cell(0);

						for (uint64_t m = (uint64_t)0; m < _core._size_2; ++m)
							for (uint64_t n = (uint64_t)0; n < _core._size_3; ++n)
								cell += _data(j + m, k + n) * _core(i, m, n);

						result(j, k, i) = std::move(cell);
					}

			return result;
		}
	}

	template <typename T, bool initialize>
	tns<T, initialize> convolute(const tns<T, initialize>& _data, const tns<T, initialize>& _core)
	{
		if (_data.is_empty() || _core.is_empty())
			return tns<T, initialize>();
		else
		{
			mtx<T, initialize> collapsed(_data._size_1, _data._size_2);
			uint64_t k = (uint64_t)0;

			for (uint64_t i = (uint64_t)0; i < collapsed._size; ++i)
			{
				T cell(0);

				for (uint64_t j = (uint64_t)0; j < _data._size_3; ++j, ++k)
					cell += _data._data[k];

				collapsed._data[i] = std::move(cell);
			}

			return convolute(collapsed, _core);
		}
	}

	template <typename T, bool initialize>
	mtx<T, initialize> convolute_n_collapse(const mtx<T, initialize>& _data, const tns<T, initialize>& _core)
	{
		if (_data.is_empty() || _core.is_empty())
			return tns<T, initialize>();
		else
		{
			mtx<T, initialize> result(_data._size_1 - _core._size_2 + (uint64_t)1,
				_data._size_2 - _core._size_3 + (uint64_t)1);

			for (uint64_t i = (uint64_t)0; i < result._size; ++i)
				result._data[i] = move(T(0));

			for (uint64_t i = (uint64_t)0; i < _core._size_1; ++i)
				for (uint64_t j = (uint64_t)0; j < result._size_1; ++j)
					for (uint64_t k = (uint64_t)0; k < result._size_2; ++k)
					{
						T cell(0);

						for (uint64_t m = (uint64_t)0; m < _core._size_2; ++m)
							for (uint64_t n = (uint64_t)0; n < _core._size_3; ++n)
								cell += _data(j + m, k + n) * _core(i, m, n);

						result(j, k) += cell;
					}

			return result;
		}
	}

	template <typename T, bool initialize>
	mtx<T, initialize> convolute_n_collapse(const tns<T, initialize>& _data, const tns<T, initialize>& _core)
	{
		if (_data.is_empty() || _core.is_empty())
			return tns<T, initialize>();
		else
		{
			mtx<T, initialize> collapsed(_data._size_1, _data._size_2);
			uint64_t k = (uint64_t)0;

			for (uint64_t i = (uint64_t)0; i < collapsed._size; ++i)
			{
				T cell(0);

				for (uint64_t j = (uint64_t)0; j < _data._size_3; ++j, ++k)
					cell += _data._data[k];

				collapsed._data[i] = std::move(cell);
			}

			return convolute_n_collapse(collapsed, _core);
		}
	}

	template <typename T, bool initialize>
	tns<T, initialize> rotate(const tns<T, initialize>& _core)
	{
		tns<T, initialize> result(_core);
		rotate(result);
		return result;
	}

	template <typename T, bool initialize>
	void rotate(tns<T, initialize>& _core)
	{
		for (uint64_t i = (uint64_t)0; i < _core._size_1; ++i)
			for (uint64_t j = (uint64_t)0; j < _core._size_2; ++j)
				for (uint64_t k = (uint64_t)0; k < _core._size_3 / (uint64_t)2; ++k)
					std::swap(_core(i, j, k), _core(i, _core._size_2 - j - (uint64_t)1, _core._size_3 - k - (uint64_t)1));
	}

	template <typename T, bool initialize>
	vec<T, initialize> operator*(const mtx<T, initialize>& _mtx, const vec<T, initialize>& _vec)
	{
		if (_vec._size == _mtx._size_2)
		{
			uint64_t size_1 = _mtx._size_1, size_2 = _mtx._size_2;
			vec<T, initialize> result(size_1);

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
			{
				T cell(0);

				for (uint64_t j = (uint64_t)0, k = i * size_2; j < size_2; ++j, ++k)
					cell += _vec._data[j] * _mtx._data[k];

				result._data[i] = std::move(cell);
			}

			return result;
		}
		else
			throw std::exception(error_msg::mtx_vec_sizes_error);
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
				T cell(0);

				for (uint64_t j = (uint64_t)0, k = i; j < size_1; ++j, k += size_2)
					cell += _vec._data[j] * _mtx._data[k];

				result._data[i] = std::move(cell);
			}

			return result;
		}
		else
			throw std::exception(error_msg::mtx_vec_sizes_error);
	}
}