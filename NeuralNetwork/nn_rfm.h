#pragma once

#include "data.h"
#include "tns.h"
#include "mtx.h"
#include "vec.h"

namespace arithmetic
{
	template <typename T, bool initialize>
	void collapse(const tns<T, initialize>&_data,vec<T, initialize>&_res)
	{
		if (_data.is_empty() || _res.is_empty())
			return;
		else if (_data.get_size_1() == _res.get_size())
		{
			uint64_t size_1 = _data.get_size_1(), size_2_3 = _data.get_size_2() * _data.get_size_3();

			for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
			{
				T cell(0);

				for (uint64_t k = (uint64_t)0; k < size_2_3; ++j, ++k)
					cell += _data(j);

				_res(i) = std::move(cell);
			}
		}
		else
			throw std::exception(error_msg::tns_vec_sizes_error);
	}

	template <typename T, bool initialize>
	tns<T, initialize> convolute(const tns<T, initialize>&_data,const tns<T, initialize>&_core)
	{
		if (_data.is_empty() || _core.is_empty())
			return tns<T, initialize>();
		else if (_data.get_size_2() >= _core.get_size_2() && _data.get_size_3() >= _core.get_size_3()
			&& _core.get_size_1() % _data.get_size_1() == (uint64_t)0)
		{
			uint64_t size_0 = _data.get_size_1();
			uint64_t size_1 = _core.get_size_1() / _data.get_size_1();
			uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
			uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

			uint64_t size_4 = _core.get_size_2();
			uint64_t size_5 = _core.get_size_3();

			tns<T, initialize> result(size_1, size_2, size_3);
			result = T(0);

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
				for (uint64_t ii = (uint64_t)0; ii < size_0; ++ii)
					for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
						for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk)
						{
							uint64_t index = &result(i, j, k) - &result((uint64_t)0);

							for (uint64_t m = j, ll = (i*size_0+ii)*size_4*size_5; m < jj; ++m)
								for (uint64_t n = k; n < kk; ++n, ++ll)
									result(index) += _data(ii, m, n) * _core(ll);
						}

			return result;
		}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	void convolute_bwd(const tns<T, initialize>&_data,const tns<T, initialize>&_core,
		const vec<bool, initialize>&_drop,tns<T, initialize>&_res)
	{
		if (_data.is_empty() || _core.is_empty() || _drop.is_empty() || _res.is_empty())
			return;
		else if(_drop.get_size()==_res.get_size_1())
		{
			uint64_t size_0 = _data.get_size_1();
			uint64_t size_1 = _core.get_size_1() / _data.get_size_1();
			uint64_t size_2 = _data.get_size_2() + _core.get_size_2() - (uint64_t)1;
			uint64_t size_3 = _data.get_size_3() + _core.get_size_3() - (uint64_t)1;

			uint64_t size_4 = _core.get_size_2();
			uint64_t size_5 = _core.get_size_3();

			int64_t size_6 = (int64_t)_data.get_size_2();
			int64_t size_7 = (int64_t)_data.get_size_3();

			if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
			{
				_res = T(0);

				for (uint64_t i = (uint64_t)0; i < size_0; ++i)
					for (uint64_t ii = (uint64_t)0; ii < size_1; ++ii)
						if(_drop(ii))
						for (int64_t j = (int64_t)1 - (int64_t)size_4, jj = (int64_t)0; jj < (int64_t)size_2; ++j, ++jj)
							for (int64_t k = (int64_t)1 - (int64_t)size_5, kk = (int64_t)0; kk < (int64_t)size_3; ++k, ++kk)
							{
								uint64_t index = &_res(ii, (uint64_t)jj, (uint64_t)kk) - &_res((uint64_t)0);
								uint64_t ll = (i * size_1+ii + (uint64_t)1) * size_4 * size_5 - (uint64_t)1;

								for (int64_t m = j; m <= jj; ++m)
									for (int64_t n = k; n <= kk; ++n, --ll)
										if (m >= (int64_t)0 && m < size_6 && n >= (int64_t)0 && n < size_7)
										_res(index) += _data(i, (uint64_t)m, (uint64_t)n) * _core(ll);
							}
			}
			else
				throw std::exception(error_msg::tns_sizes_error);
		}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	void convolute_fwd(const tns<T, initialize>&_data,const tns<T, initialize>&_core,
		const vec<bool, initialize>&_drop,tns<T, initialize>&_res)
	{
		if (_data.is_empty() || _core.is_empty() || _drop.is_empty() || _res.is_empty())
			return;
		else if (_data.get_size_2() >= _core.get_size_2() && _data.get_size_3() >= _core.get_size_3()
			&&_drop.get_size()==_res.get_size_1())
		{
			uint64_t size_0 = _data.get_size_1();
			uint64_t size_1 = _core.get_size_1() / _data.get_size_1();
			uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
			uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

			uint64_t size_4 = _core.get_size_2();
			uint64_t size_5 = _core.get_size_3();

			if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
			{
				_res = T(0);

				for (uint64_t i = (uint64_t)0; i < size_1; ++i)
					if(_drop(i))
					for (uint64_t ii = (uint64_t)0; ii < size_0; ++ii)
						for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
							for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk)
							{
								uint64_t index = &_res(i, j, k) - &_res((uint64_t)0);

								for (uint64_t m = j, ll = (i*size_0+ii)*size_4*size_5; m < jj; ++m)
									for (uint64_t n = k; n < kk; ++n, ++ll)
										_res(index) += _data(ii, m, n) * _core(ll);
							}
			}
			else
				throw std::exception(error_msg::tns_sizes_error);
		}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	void convolute_rev(const tns<T, initialize>&_data,const tns<T, initialize>&_core,tns<T, initialize>&_res)
	{
		if (_data.is_empty() || _core.is_empty() || _res.is_empty())
			return;
		else if (_data.get_size_2() >= _core.get_size_2() && _data.get_size_3() >= _core.get_size_3())
		{
			uint64_t size_1 = _data.get_size_1() * _core.get_size_1();
			uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
			uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

			uint64_t size_4 = _core.get_size_2();
			uint64_t size_5 = _core.get_size_3();
			
			uint64_t size_6 = _core.get_size_1();
			uint64_t size_7 = _data.get_size_1();

			if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
			{
				_res = T(0);

				for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_6; ++i)
					for (uint64_t ii = (uint64_t)0; ii < size_7; ++ii)
						for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
							for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
								for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
									for (uint64_t n = k; n < kk; ++n, ++ll)
										_res(l) += _data(ii, m, n) * _core(ll);

				_res/=T(size_2*size_3*size_4*size_5);
			}
			else
				throw std::exception(error_msg::tns_sizes_error);
		}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	void convolute_rev(const tns<T, initialize>&_data,const vec<T, initialize>&_core,vec<T, initialize>&_res)
	{
		if (_data.is_empty() || _core.is_empty() || _res.is_empty())
			return;
		else
		{
			uint64_t size = _core.get_size();

			if (_res.get_size() == size)
				_res = _core;
			else
				throw std::exception(error_msg::vec_sizes_error);
		}
	}

	template <typename T, bool initialize>
	vec<T, initialize> multiply(const mtx<T, initialize>&_mtx,const vec<T, initialize>&_vec)
	{
		if (_mtx.is_empty() || _vec.is_empty())
			return vec<T, initialize>();
		else if (_mtx.get_size_2() == _vec.get_size())
		{
			uint64_t size_1 = _mtx.get_size_1(), size_2 = _mtx.get_size_2();
			vec<T, initialize> result(size_1);

			for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
			{
				T cell(0);

				for (uint64_t k = (uint64_t)0; k < size_2; ++j, ++k)
					cell += _mtx(j) * _vec(k);

				result(i) = std::move(cell);
			}

			return result;
		}
		else
			throw std::exception(error_msg::mtx_vec_sizes_error);
	}

	template <typename T, bool initialize>
	void multiply_bwd(const mtx<T, initialize>&_mtx,const vec<T, initialize>&_vec,
		const vec<bool, initialize>&_drop,vec<T, initialize>&_res)
	{
		if (_mtx.is_empty() || _vec.is_empty() || _drop.is_empty() || _res.is_empty())
			return;
		else if (_mtx.get_size_1() == _vec.get_size() && _mtx.get_size_2() == _res.get_size()
			&& _drop.get_size()==_res.get_size())
		{
			uint64_t size_1 = _mtx.get_size_1(), size_2 = _mtx.get_size_2();
			_res = T(0);

			for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
				for (uint64_t k = (uint64_t)0; k < size_2; ++j, ++k)
					if(_drop(k))
					_res(k) += _mtx(j) * _vec(i);
		}
		else
			throw std::exception(error_msg::mtx_vec_sizes_error);
	}

	template <typename T, bool initialize>
	void multiply_fwd(const mtx<T, initialize>&_mtx,const vec<T, initialize>&_vec,
		const vec<bool, initialize>&_drop,vec<T, initialize>&_res)
	{
		if (_mtx.is_empty() || _vec.is_empty() || _drop.is_empty() || _res.is_empty())
			return;
		else if (_mtx.get_size_2() == _vec.get_size() && _mtx.get_size_1() == _res.get_size()
			&& _drop.get_size()==_res.get_size())
		{
			uint64_t size_1 = _mtx.get_size_1(), size_2 = _mtx.get_size_2();

			for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
				if(_drop(i))
			{
				T cell(0);

				for (uint64_t k = (uint64_t)0; k < size_2; ++j, ++k)
					cell += _mtx(j) * _vec(k);

				_res(i) = std::move(cell);
			}
				else
				{
					j+=size_2;
					_res(i) =std::move(T(0));
				}
		}
		else
			throw std::exception(error_msg::mtx_vec_sizes_error);
	}

	template <typename T, bool initialize>
	tns<T, initialize> pool(const tns<T, initialize>&_data)
	{
		if (_data.is_empty())
			return tns<T, initialize>();
		else
		{
			uint64_t size_1 = _data.get_size_1(), size_2 = _data.get_size_2(), size_3 = _data.get_size_3();
			tns<T, initialize> result(size_1, size_2 / (uint64_t)2, size_3 / (uint64_t)2);

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
				for (uint64_t j = (uint64_t)0, jj = (uint64_t)0; jj < size_2; j += jj % (uint64_t)2, ++jj)
					for (uint64_t k = (uint64_t)0, kk = (uint64_t)0; kk < size_3; k += kk % (uint64_t)2, ++kk)
						if ((jj | kk) % (uint64_t)2 == (uint64_t)0 || result(i, j, k) < _data(i, jj, kk))
							result(i, j, k) = _data(i, jj, kk);

			return result;
		}
	}

	template <typename T, bool initialize>
	void pool_bwd(const tns<T, initialize>&_data,const tns<uint8_t, initialize>&_pool,tns<T, initialize>&_res)
	{
		if (_data.is_empty() || _pool.is_empty() || _res.is_empty())
			return;
		else if (_data.get_size_1() == _pool.get_size_1() && _data.get_size_1() == _res.get_size_1()
			&& _data.get_size_2() == _pool.get_size_2() && _data.get_size_2() * (uint64_t)2 == _res.get_size_2()
			&& _data.get_size_3() == _pool.get_size_3() && _data.get_size_3() * (uint64_t)2 == _res.get_size_3())
		{
			uint64_t size_1 = _data.get_size_1(), size_2 = _data.get_size_2(), size_3 = _data.get_size_3();
			_res = T(0);

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
				for (uint64_t j = (uint64_t)0, jj = (uint64_t)0; jj < size_2; j += (uint64_t)2, ++jj)
					for (uint64_t k = (uint64_t)0, kk = (uint64_t)0; kk < size_3; k += (uint64_t)2, ++kk)
					{
						uint64_t index = (uint64_t)_pool(i, jj, kk);
						_res(i, j + (index / (uint64_t)2), k + (index % (uint64_t)2)) = _data(i, jj, kk);
					}
		}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	void pool_fwd(const tns<T, initialize>&_data,tns<uint8_t, initialize>&_pool,tns<T, initialize>&_res)
	{
		if (_data.is_empty() || _pool.is_empty() || _res.is_empty())
			return;
		else if (_res.get_size_1() == _pool.get_size_1() && _res.get_size_1() == _data.get_size_1()
			&& _res.get_size_2() == _pool.get_size_2() && _res.get_size_2() * (uint64_t)2 == _data.get_size_2()
			&& _res.get_size_3() == _pool.get_size_3() && _res.get_size_3() * (uint64_t)2 == _data.get_size_3())
		{
			uint64_t size_1 = _data.get_size_1(), size_2 = _data.get_size_2(), size_3 = _data.get_size_3();

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
				for (uint64_t j = (uint64_t)0, jj = (uint64_t)0; jj < size_2; j += jj % (uint64_t)2, ++jj)
					for (uint64_t k = (uint64_t)0, kk = (uint64_t)0; kk < size_3; k += kk % (uint64_t)2, ++kk)
						if ((jj | kk) % (uint64_t)2 == (uint64_t)0 || _res(i, j, k) < _data(i, jj, kk))
						{
							_pool(i, j, k) = (uint8_t)((jj % (uint64_t)2) * (uint64_t)2 + (kk % (uint64_t)2));
							_res(i, j, k) = _data(i, jj, kk);
						}
		}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	vec<T, initialize> pool_full(const tns<T, initialize>&_data,bool _max_pool)
	{
		if (_data.is_empty())
			return vec<T, initialize>();
		else
		{
			uint64_t size_1 = _data.get_size_1(), size_2_3 = _data.get_size_2() * _data.get_size_3();
			vec<T, initialize> result(size_1);

			if (_max_pool)
				for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
				{
					T cell = _data(j);
					++j;

					for (uint64_t k = (uint64_t)1; k < size_2_3; ++j, ++k)
						if (cell < _data(j))
							cell = _data(j);

					result(i) = std::move(cell);
				}
			else
				for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
				{
					T cell(0);

					for (uint64_t k = (uint64_t)0; k < size_2_3; ++j, ++k)
						cell += _data(j);

					result(i) = std::move(cell / T(size_2_3));
				}

			return result;
		}
	}

	template <typename T, bool initialize>
	void pool_full_bwd(const vec<T, initialize>&_data,const vec<uint64_t, initialize>&_pool,
		tns<T, initialize>&_res,bool _max_pool)
	{
		if (_data.is_empty() || (_pool.is_empty() && _max_pool) || _res.is_empty())
			return;
		else if ((!_max_pool || _res.get_size_1() == _pool.get_size()) && _res.get_size_1() == _data.get_size())
		{
			uint64_t size_1 = _res.get_size_1(), size_2_3 = _res.get_size_2() * _res.get_size_3();

			if (_max_pool)
			{
				_res = T(0);

				for (uint64_t i = (uint64_t)0; i < size_1; ++i)
					_res(_pool(i)) = _data(i);
			}
			else
				for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
					for (uint64_t k = (uint64_t)0; k < size_2_3; ++j, ++k)
						_res(j) = _data(i);
		}
		else
			throw std::exception(error_msg::tns_vec_sizes_error);
	}

	template <typename T, bool initialize>
	void pool_full_fwd(const tns<T, initialize>&_data,vec<uint64_t, initialize>&_pool,
		vec<T, initialize>&_res,bool _max_pool)
	{
		if (_data.is_empty() || (_pool.is_empty() && _max_pool) || _res.is_empty())
			return;
		else if ((!_max_pool || _data.get_size_1() == _pool.get_size()) && _data.get_size_1() == _res.get_size())
		{
			uint64_t size_1 = _data.get_size_1(), size_2_3 = _data.get_size_2() * _data.get_size_3();

			if (_max_pool)
				for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
				{
					uint64_t index = j;
					T cell = _data(j);
					++j;

					for (uint64_t k = (uint64_t)1; k < size_2_3; ++j, ++k)
						if (cell < _data(j))
						{
							index = j;
							cell = _data(j);
						}

					_pool(i) = index;
					_res(i) = std::move(cell);
				}
			else
				for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
				{
					T cell(0);

					for (uint64_t k = (uint64_t)0; k < size_2_3; ++j, ++k)
						cell += _data(j);

					_res(i) = std::move(cell / T(size_2_3));
				}
		}
		else
			throw std::exception(error_msg::tns_vec_sizes_error);
	}
}