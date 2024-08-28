#pragma once

#include "data.h"
#include "tns.h"
#include "mtx.h"
#include "vec.h"
#include "nn_params.h"

namespace arithmetic
{
	template <typename T, bool initialize>
	void collapse(const tns<T, initialize>& _data, vec<T, initialize>& _res)
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
	tns<T, initialize> convolute(const tns<T, initialize>& _data, const tns<T, initialize>& _core, nn_params::nn_convo_t _type)
	{
		if (_data.is_empty() || _core.is_empty())
			return tns<T, initialize>();
		else if (_data.get_size_2() >= _core.get_size_2() && _data.get_size_3() >= _core.get_size_3())
			switch (_type)
			{
			case nn_params::nn_convo_t::many_to_many:
			{
				uint64_t size_1 = _data.get_size_1() * _core.get_size_1();
				uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
				uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

				uint64_t size_4 = _core.get_size_2();
				uint64_t size_5 = _core.get_size_3();

				uint64_t size_6 = _core.get_size_1();
				uint64_t size_7 = _data.get_size_1();

				tns<T, initialize> result(size_1, size_2, size_3);

				for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_6; ++i)
					for (uint64_t ii = (uint64_t)0; ii < size_7; ++ii)
						for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
							for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
							{
								T cell(0);

								for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
									for (uint64_t n = k; n < kk; ++n, ++ll)
										cell += _data(ii, m, n) * _core(ll);

								result(l) = std::move(cell);
							}

				return result;
			}

			case nn_params::nn_convo_t::many_to_one:
			{
				uint64_t size_1 = _core.get_size_1();
				uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
				uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

				uint64_t size_4 = _core.get_size_2();
				uint64_t size_5 = _core.get_size_3();

				tns<T, initialize> result(size_1, size_2, size_3);
				const tns<T, initialize>* colla = nullptr;
				tns<T, initialize> colla_man;

				if (_data.get_size_1() == (uint64_t)1)
					colla = &_data;
				else
				{
					colla_man = std::move(tns<T, initialize>((uint64_t)1, _data.get_size_2(), _data.get_size_3()));
					colla = &colla_man;

					uint64_t size_data = _data.get_size();
					uint64_t size_colla = colla_man.get_size();

					for (uint64_t i = (uint64_t)0; i < size_colla; ++i)
						colla_man(i) = _data(i);

					for (uint64_t i = (uint64_t)0, j = size_colla; j < size_data; ++i, ++j)
					{
						if (i == size_colla)
							i = (uint64_t)0;

						colla_man(i) += _data(j);
					}

					colla_man /= T(_data.get_size_1());
				}

				__assume(colla != nullptr);
				for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_1; ++i)
					for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
						for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
						{
							T cell(0);

							for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
								for (uint64_t n = k; n < kk; ++n, ++ll)
									cell += (*colla)((uint64_t)0, m, n) * _core(ll);

							result(l) = std::move(cell);
						}

				return result;
			}

			case nn_params::nn_convo_t::one_to_one:
			{
				if (_data.get_size_1() == _core.get_size_1())
				{
					uint64_t size_1 = _data.get_size_1();
					uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
					uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

					uint64_t size_4 = _core.get_size_2();
					uint64_t size_5 = _core.get_size_3();

					tns<T, initialize> result(size_1, size_2, size_3);

					for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_1; ++i)
						for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
							for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
							{
								T cell(0);

								for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
									for (uint64_t n = k; n < kk; ++n, ++ll)
										cell += _data(i, m, n) * _core(ll);

								result(l) = std::move(cell);
							}

					return result;
				}
				else
					throw std::exception(error_msg::tns_sizes_error);
			}

			default:
			{
				throw std::exception(error_msg::convo_wrong_type);
			}
			}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	void convolute_bwd(const tns<T, initialize>& _data, const tns<T, initialize>& _core,
		tns<T, initialize>& _res, nn_params::nn_convo_t _type)
	{
		if (_data.is_empty() || _core.is_empty() || _res.is_empty())
			return;
		else
			switch (_type)
			{
			case nn_params::nn_convo_t::many_to_many:
			{
				uint64_t size_1 = _data.get_size_1() / _core.get_size_1();
				uint64_t size_2 = _data.get_size_2() + _core.get_size_2() - (uint64_t)1;
				uint64_t size_3 = _data.get_size_3() + _core.get_size_3() - (uint64_t)1;

				uint64_t size_4 = _core.get_size_2();
				uint64_t size_5 = _core.get_size_3();

				uint64_t size_6 = _data.get_size_1();
				int64_t size_7 = (int64_t)_data.get_size_2();
				int64_t size_8 = (int64_t)_data.get_size_3();

				uint64_t size_9 = _res.get_size();

				if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
				{
					_res = T(0);

					for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_6; ++i)
						for (int64_t j = (int64_t)1 - (int64_t)size_4, jj = (int64_t)0; jj < (int64_t)size_2; ++j, ++jj)
							for (int64_t k = (int64_t)1 - (int64_t)size_5, kk = (int64_t)0; kk < (int64_t)size_3; ++k, ++kk, ++l)
							{
								if (l == size_9)
									l = (uint64_t)0;

								T cell(0);

								uint64_t ll = (i / size_1) * size_4 * size_5;
								for (int64_t m = j; m <= jj; ++m)
									for (int64_t n = k; n <= kk; ++n, ++ll)
										if (m >= (int64_t)0 && m < size_7 && n >= (int64_t)0 && n < size_8)
											cell += _data(i, (uint64_t)m, (uint64_t)n) * _core(ll);

								_res(l) += std::move(cell);
							}
				}
				else
					throw std::exception(error_msg::tns_sizes_error);

				break;
			}

			case nn_params::nn_convo_t::many_to_one:
			{
				if (_data.get_size_1() == _core.get_size_1())
				{
					uint64_t size_1 = _res.get_size_1();
					uint64_t size_2 = _data.get_size_2() + _core.get_size_2() - (uint64_t)1;
					uint64_t size_3 = _data.get_size_3() + _core.get_size_3() - (uint64_t)1;

					uint64_t size_4 = _core.get_size_2();
					uint64_t size_5 = _core.get_size_3();

					uint64_t size_6 = size_2 * size_3;

					uint64_t size_7 = _data.get_size_1();
					int64_t size_8 = (int64_t)_data.get_size_2();
					int64_t size_9 = (int64_t)_data.get_size_3();

					uint64_t size_10 = _res.get_size();

					if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
					{
						for (uint64_t i = (uint64_t)0; i < size_6; ++i)
							_res(i) = std::move(T(0));

						for (uint64_t i = (uint64_t)0; i < size_7; ++i)
						{
							uint64_t l = (uint64_t)0;
							for (int64_t j = (int64_t)1 - (int64_t)size_4, jj = (int64_t)0; jj < (int64_t)size_2; ++j, ++jj)
								for (int64_t k = (int64_t)1 - (int64_t)size_5, kk = (int64_t)0; kk < (int64_t)size_3; ++k, ++kk, ++l)
								{
									T cell(0);

									uint64_t ll = i * size_4 * size_5;
									for (int64_t m = j; m <= jj; ++m)
										for (int64_t n = k; n <= kk; ++n, ++ll)
											if (m >= (int64_t)0 && m < size_8 && n >= (int64_t)0 && n < size_9)
												cell += _data(i, (uint64_t)m, (uint64_t)n) * _core(ll);

									_res(l) += std::move(cell);
								}
						}

						T size_7_res(size_7);

						for (uint64_t i = (uint64_t)0; i < size_6; ++i)
							_res(i) /= size_7_res;

						for (uint64_t i = (uint64_t)0, j = size_6; j < size_10; ++i, ++j)
						{
							if (i == size_6)
								i = (uint64_t)0;

							_res(j) = _res(i);
						}
					}
					else
						throw std::exception(error_msg::tns_sizes_error);
				}
				else
					throw std::exception(error_msg::tns_sizes_error);

				break;
			}

			case nn_params::nn_convo_t::one_to_one:
			{
				if (_data.get_size_1() == _core.get_size_1())
				{
					uint64_t size_1 = _data.get_size_1();
					uint64_t size_2 = _data.get_size_2() + _core.get_size_2() - (uint64_t)1;
					uint64_t size_3 = _data.get_size_3() + _core.get_size_3() - (uint64_t)1;

					uint64_t size_4 = _core.get_size_2();
					uint64_t size_5 = _core.get_size_3();

					int64_t size_6 = (int64_t)_data.get_size_2();
					int64_t size_7 = (int64_t)_data.get_size_3();

					if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
						for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_1; ++i)
							for (int64_t j = (int64_t)1 - (int64_t)size_4, jj = (int64_t)0; jj < (int64_t)size_2; ++j, ++jj)
								for (int64_t k = (int64_t)1 - (int64_t)size_5, kk = (int64_t)0; kk < (int64_t)size_3; ++k, ++kk, ++l)
								{
									T cell(0);

									uint64_t ll = i * size_4 * size_5;
									for (int64_t m = j; m <= jj; ++m)
										for (int64_t n = k; n <= kk; ++n, ++ll)
											if (m >= (int64_t)0 && m < size_6 && n >= (int64_t)0 && n < size_7)
												cell += _data(i, (uint64_t)m, (uint64_t)n) * _core(ll);

									_res(l) = std::move(cell);
								}
					else
						throw std::exception(error_msg::tns_sizes_error);
				}
				else
					throw std::exception(error_msg::tns_sizes_error);

				break;
			}

			default:
			{
				throw std::exception(error_msg::convo_wrong_type);
			}
			}
	}

	template <typename T, bool initialize>
	void convolute_fwd(const tns<T, initialize>& _data, const tns<T, initialize>& _core,
		tns<T, initialize>& _res, nn_params::nn_convo_t _type)
	{
		if (_data.is_empty() || _core.is_empty() || _res.is_empty())
			return;
		else if (_data.get_size_2() >= _core.get_size_2() && _data.get_size_3() >= _core.get_size_3())
			switch (_type)
			{
			case nn_params::nn_convo_t::many_to_many:
			{
				uint64_t size_1 = _data.get_size_1() * _core.get_size_1();
				uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
				uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

				uint64_t size_4 = _core.get_size_2();
				uint64_t size_5 = _core.get_size_3();

				uint64_t size_6 = _core.get_size_1();
				uint64_t size_7 = _data.get_size_1();

				if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
					for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_6; ++i)
						for (uint64_t ii = (uint64_t)0; ii < size_7; ++ii)
							for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
								for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
								{
									T cell(0);

									for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
										for (uint64_t n = k; n < kk; ++n, ++ll)
											cell += _data(ii, m, n) * _core(ll);

									_res(l) = std::move(cell);
								}
				else
					throw std::exception(error_msg::tns_sizes_error);

				break;
			}

			case nn_params::nn_convo_t::many_to_one:
			{
				uint64_t size_1 = _core.get_size_1();
				uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
				uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

				uint64_t size_4 = _core.get_size_2();
				uint64_t size_5 = _core.get_size_3();

				if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
				{
					const tns<T, initialize>* colla = nullptr;
					tns<T, initialize> colla_man;

					if (_data.get_size_1() == (uint64_t)1)
						colla = &_data;
					else
					{
						colla_man = std::move(tns<T, initialize>((uint64_t)1, _data.get_size_2(), _data.get_size_3()));
						colla = &colla_man;

						uint64_t size_data = _data.get_size();
						uint64_t size_colla = colla_man.get_size();

						for (uint64_t i = (uint64_t)0; i < size_colla; ++i)
							colla_man(i) = _data(i);

						for (uint64_t i = (uint64_t)0, j = size_colla; j < size_data; ++i, ++j)
						{
							if (i == size_colla)
								i = (uint64_t)0;

							colla_man(i) += _data(j);
						}

						colla_man /= T(_data.get_size_1());
					}

					__assume(colla != nullptr);
					for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_1; ++i)
						for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
							for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
							{
								T cell(0);

								for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
									for (uint64_t n = k; n < kk; ++n, ++ll)
										cell += (*colla)((uint64_t)0, m, n) * _core(ll);

								_res(l) = std::move(cell);
							}
				}
				else
					throw std::exception(error_msg::tns_sizes_error);

				break;
			}

			case nn_params::nn_convo_t::one_to_one:
			{
				if (_data.get_size_1() == _core.get_size_1())
				{
					uint64_t size_1 = _data.get_size_1();
					uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
					uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

					uint64_t size_4 = _core.get_size_2();
					uint64_t size_5 = _core.get_size_3();

					if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
						for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_1; ++i)
							for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
								for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
								{
									T cell(0);

									for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
										for (uint64_t n = k; n < kk; ++n, ++ll)
											cell += _data(i, m, n) * _core(ll);

									_res(l) = std::move(cell);
								}
					else
						throw std::exception(error_msg::tns_sizes_error);
				}
				else
					throw std::exception(error_msg::tns_sizes_error);

				break;
			}

			default:
			{
				throw std::exception(error_msg::convo_wrong_type);
			}
			}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	void convolute_rev(const tns<T, initialize>& _data, const tns<T, initialize>& _core,
		tns<T, initialize>& _res, nn_params::nn_convo_t _type)
	{
		if (_data.is_empty() || _core.is_empty() || _res.is_empty())
			return;
		else if (_data.get_size_2() >= _core.get_size_2() && _data.get_size_3() >= _core.get_size_3())
			switch (_type)
			{
			case nn_params::nn_convo_t::many_to_many:
			{
				uint64_t size_1 = _core.get_size_1() / _data.get_size_1();
				uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
				uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

				uint64_t size_4 = _core.get_size_2();
				uint64_t size_5 = _core.get_size_3();

				if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
				{
					const tns<T, initialize>* colla_core = nullptr, * colla_data = nullptr;
					tns<T, initialize> colla_mc, colla_md;

					if (_data.get_size_1() == (uint64_t)1)
					{
						colla_core = &_core;
						colla_data = &_data;
					}
					else
					{
						colla_mc = std::move(tns<T, initialize>(size_1, _core.get_size_2(), _core.get_size_3()));
						colla_core = &colla_mc;

						colla_md = std::move(tns<T, initialize>((uint64_t)1, _data.get_size_2(), _data.get_size_3()));
						colla_data = &colla_md;

						uint64_t size_colla_c = colla_mc.get_size_2() * colla_mc.get_size_3();
						uint64_t size_colla_d = colla_md.get_size();
						uint64_t size_data = _data.get_size();

						for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
						{
							uint64_t start_colla = i * size_colla_c, end_colla = (i + (uint64_t)1) * size_colla_c;

							for (uint64_t k = start_colla; k < end_colla; ++j, ++k)
								colla_mc(k) = _core(j);

							for (uint64_t k = (uint64_t)1; k < _data.get_size_1(); ++k)
								for (uint64_t l = start_colla; l < end_colla; ++j, ++l)
									colla_mc(l) += _core(j);
						}

						for (uint64_t i = (uint64_t)0; i < size_colla_d; ++i)
							colla_md(i) = _data(i);

						for (uint64_t i = (uint64_t)0, j = size_colla_d; j < size_data; ++i, ++j)
						{
							if (i == size_colla_d)
								i = (uint64_t)0;

							colla_md(i) += _data(j);
						}

						colla_mc /= T(_data.get_size_1());
						colla_md /= T(_data.get_size_1());
					}

					__assume(colla_core != nullptr);
					__assume(colla_data != nullptr);
					for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_1; ++i)
						for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
							for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
							{
								T cell(0);

								for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
									for (uint64_t n = k; n < kk; ++n, ++ll)
										cell += (*colla_data)((uint64_t)0, m, n) * (*colla_core)(ll);

								_res(l) = std::move(cell);
							}
				}
				else
					throw std::exception(error_msg::tns_sizes_error);

				break;
			}

			case nn_params::nn_convo_t::many_to_one:
			{
				uint64_t size_1 = _core.get_size_1();
				uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
				uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

				uint64_t size_4 = _core.get_size_2();
				uint64_t size_5 = _core.get_size_3();

				if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
				{
					const tns<T, initialize>* colla = nullptr;
					tns<T, initialize> colla_man;

					if (_data.get_size_1() == (uint64_t)1)
						colla = &_data;
					else
					{
						colla_man = std::move(tns<T, initialize>((uint64_t)1, _data.get_size_2(), _data.get_size_3()));
						colla = &colla_man;

						uint64_t size_data = _data.get_size();
						uint64_t size_colla = colla_man.get_size();

						for (uint64_t i = (uint64_t)0; i < size_colla; ++i)
							colla_man(i) = _data(i);

						for (uint64_t i = (uint64_t)0, j = size_colla; j < size_data; ++i, ++j)
						{
							if (i == size_colla)
								i = (uint64_t)0;

							colla_man(i) += _data(j);
						}

						colla_man /= T(_data.get_size_1());
					}

					__assume(colla != nullptr);
					for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_1; ++i)
						for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
							for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
							{
								T cell(0);

								for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
									for (uint64_t n = k; n < kk; ++n, ++ll)
										cell += (*colla)((uint64_t)0, m, n) * _core(ll);

								_res(l) = std::move(cell);
							}
				}
				else
					throw std::exception(error_msg::tns_sizes_error);

				break;
			}

			case nn_params::nn_convo_t::one_to_one:
			{
				if (_data.get_size_1() == _core.get_size_1())
				{
					uint64_t size_1 = _data.get_size_1();
					uint64_t size_2 = _data.get_size_2() - _core.get_size_2() + (uint64_t)1;
					uint64_t size_3 = _data.get_size_3() - _core.get_size_3() + (uint64_t)1;

					uint64_t size_4 = _core.get_size_2();
					uint64_t size_5 = _core.get_size_3();

					if (_res.get_size_1() == size_1 && _res.get_size_2() == size_2 && _res.get_size_3() == size_3)
						for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_1; ++i)
							for (uint64_t j = (uint64_t)0, jj = size_4; j < size_2; ++j, ++jj)
								for (uint64_t k = (uint64_t)0, kk = size_5; k < size_3; ++k, ++kk, ++l)
								{
									T cell(0);

									for (uint64_t m = j, ll = i * size_4 * size_5; m < jj; ++m)
										for (uint64_t n = k; n < kk; ++n, ++ll)
											cell += _data(i, m, n) * _core(ll);

									_res(l) = std::move(cell);
								}
					else
						throw std::exception(error_msg::tns_sizes_error);
				}
				else
					throw std::exception(error_msg::tns_sizes_error);

				break;
			}

			default:
			{
				throw std::exception(error_msg::convo_wrong_type);
			}
			}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	void convolute_rev_colla(const tns<T, initialize>& _data, const vec<T, initialize>& _core,
		vec<T, initialize>& _res, nn_params::nn_convo_t _type)
	{
		if (_data.is_empty() || _res.is_empty())
			return;
		else
			switch (_type)
			{
			case nn_params::nn_convo_t::many_to_many:
			{
				uint64_t size = _core.get_size() / _data.get_size_1();

				if (_res.get_size() == size)
				{
					for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size; ++i)
					{
						_res(i) = _core(j);
						++j;

						for (uint64_t k = (uint64_t)1; k < _data.get_size_1(); ++k, ++j)
							_res(i) += _core(j);
					}
				}
				else
					throw std::exception(error_msg::vec_sizes_error);

				break;
			}

			case nn_params::nn_convo_t::many_to_one:
			case nn_params::nn_convo_t::one_to_one:
			{
				if (_res.get_size() == _core.get_size())
					_res = _core;
				else
					throw std::exception(error_msg::vec_sizes_error);

				break;
			}

			default:
			{
				throw std::exception(error_msg::convo_wrong_type);
			}
			}
	}

	template <typename T, bool initialize>
	vec<T, initialize> multiply(const mtx<T, initialize>& _mtx, const vec<T, initialize>& _vec)
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
	void multiply_bwd(const mtx<T, initialize>& _mtx, const vec<T, initialize>& _vec, vec<T, initialize>& _res)
	{
		if (_mtx.is_empty() || _vec.is_empty() || _res.is_empty())
			return;
		else if (_mtx.get_size_1() == _vec.get_size() && _mtx.get_size_2() == _res.get_size())
		{
			uint64_t size_1 = _mtx.get_size_1(), size_2 = _mtx.get_size_2();

			for (uint64_t i = (uint64_t)0; i < size_2; ++i)
			{
				T cell(0);

				for (uint64_t j = i, k = (uint64_t)0; k < size_1; j += size_2, ++k)
					cell += _mtx(j) * _vec(k);

				_res(i) = std::move(cell);
			}
		}
		else
			throw std::exception(error_msg::mtx_vec_sizes_error);
	}

	template <typename T, bool initialize>
	void multiply_fwd(const mtx<T, initialize>& _mtx, const vec<T, initialize>& _vec, vec<T, initialize>& _res)
	{
		if (_mtx.is_empty() || _vec.is_empty() || _res.is_empty())
			return;
		else if (_mtx.get_size_2() == _vec.get_size() && _mtx.get_size_1() == _res.get_size())
		{
			uint64_t size_1 = _mtx.get_size_1(), size_2 = _mtx.get_size_2();

			for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < size_1; ++i)
			{
				T cell(0);

				for (uint64_t k = (uint64_t)0; k < size_2; ++j, ++k)
					cell += _mtx(j) * _vec(k);

				_res(i) = std::move(cell);
			}
		}
		else
			throw std::exception(error_msg::mtx_vec_sizes_error);
	}

	template <typename T, bool initialize>
	tns<T, initialize> pool(const tns<T, initialize>& _data)
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
	void pool_bwd(const tns<T, initialize>& _data, const tns<uint8_t, initialize>& _pool, tns<T, initialize>& _res)
	{
		if (_data.is_empty() || _pool.is_empty() || _res.is_empty())
			return;
		else if (_data.get_size_1() == _pool.get_size_1() && _data.get_size_1() == _res.get_size_1()
			&& _data.get_size_2() == _pool.get_size_2() && _data.get_size_2() * (uint64_t)2 == _res.get_size_2()
			&& _data.get_size_3() == _pool.get_size_3() && _data.get_size_3() * (uint64_t)2 == _res.get_size_3())
		{
			uint64_t size_1 = _data.get_size_1(), size_2 = _data.get_size_2(), size_3 = _data.get_size_3();

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
				for (uint64_t j = (uint64_t)0, jj = (uint64_t)0; jj < size_2; j += (uint64_t)2, ++jj)
					for (uint64_t k = (uint64_t)0, kk = (uint64_t)0; kk < size_3; k += (uint64_t)2, ++kk)
					{
						uint8_t index = _pool(i, jj, kk);

						_res(i, j + (uint64_t)0, k + (uint64_t)0) = index == (uint8_t)0b00 ? _data(i, jj, kk) : std::move(T(0));
						_res(i, j + (uint64_t)0, k + (uint64_t)1) = index == (uint8_t)0b01 ? _data(i, jj, kk) : std::move(T(0));
						_res(i, j + (uint64_t)1, k + (uint64_t)0) = index == (uint8_t)0b10 ? _data(i, jj, kk) : std::move(T(0));
						_res(i, j + (uint64_t)1, k + (uint64_t)1) = index == (uint8_t)0b11 ? _data(i, jj, kk) : std::move(T(0));
					}
		}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	void pool_fwd(const tns<T, initialize>& _data, tns<uint8_t, initialize>& _pool, tns<T, initialize>& _res)
	{
		if (_data.is_empty() || _pool.is_empty() || _res.is_empty())
			return;
		else if (_pool.get_size_1() == _res.get_size_1() && _data.get_size_1() == _res.get_size_1()
			&& _pool.get_size_2() == _res.get_size_2() && _data.get_size_2() == _res.get_size_2() * (uint64_t)2
			&& _pool.get_size_3() == _res.get_size_3() && _data.get_size_3() == _res.get_size_3() * (uint64_t)2)
		{
			uint64_t size_1 = _data.get_size_1(), size_2 = _data.get_size_2(), size_3 = _data.get_size_3();

			for (uint64_t i = (uint64_t)0; i < size_1; ++i)
				for (uint64_t j = (uint64_t)0, jj = (uint64_t)0; jj < size_2; j += jj % (uint64_t)2, ++jj)
					for (uint64_t k = (uint64_t)0, kk = (uint64_t)0; kk < size_3; k += kk % (uint64_t)2, ++kk)
						if ((jj | kk) % (uint64_t)2 == (uint64_t)0 || _res(i, j, k) < _data(i, jj, kk))
						{
							_pool(i, j, k) = ((uint8_t)jj % (uint8_t)2) * (uint8_t)2 + ((uint8_t)kk % (uint8_t)2);
							_res(i, j, k) = _data(i, jj, kk);
						}
		}
		else
			throw std::exception(error_msg::tns_sizes_error);
	}

	template <typename T, bool initialize>
	vec<T, initialize> pool_full(const tns<T, initialize>& _data,bool _max_pool)
	{
		if (_data.is_empty())
			return vec<T, initialize>();
		else
		{
			uint64_t size_1 = _data.get_size_1(), size_2_3 = _data.get_size_2()*_data.get_size_3();
			vec<T, initialize> result(size_1);

			if (_max_pool)
				for (uint64_t i = (uint64_t)0,j=(uint64_t)0; i < size_1; ++i)
				{
					T cell = _data(j);
					++j;

					for (uint64_t k = (uint64_t)1; k < size_2_3; ++j,++k)
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

					result(i) = std::move(cell/T(size_2_3));
				}

			return result;
		}
	}

	template <typename T, bool initialize>
	void pool_full_bwd(const vec<T, initialize>& _data, const vec<uint64_t, initialize>& _pool,
		tns<T, initialize>& _res, bool _max_pool)
	{
		if (_data.is_empty() || (_pool.is_empty()&&_max_pool) || _res.is_empty())
			return;
		else if ((!_max_pool || _res.get_size_1() == _pool.get_size()) && _res.get_size_1() == _data.get_size())
		{
			uint64_t size_1 = _res.get_size_1(), size_2_3 = _res.get_size_2() * _res.get_size_3();

			if (_max_pool)
			{
				_res = T(0);

				for (uint64_t i = (uint64_t)0; i < size_1; ++i)
					_res(i, (uint64_t)0, _pool(i)) = _data(i);
			}
			else
				for (uint64_t i = (uint64_t)0,j=(uint64_t)0; i < size_1; ++i)
					for (uint64_t k = (uint64_t)0; k < size_2_3; ++j, ++k)
						_res(j) = _data(i);
		}
		else
			throw std::exception(error_msg::tns_vec_sizes_error);
	}

	template <typename T, bool initialize>
	void pool_full_fwd(const tns<T, initialize>& _data, vec<uint64_t, initialize>& _pool,
		vec<T, initialize>& _res, bool _max_pool)
	{
		if (_data.is_empty() || (_pool.is_empty()&&_max_pool) || _res.is_empty())
			return;
		else if ((!_max_pool||_data.get_size_1()==_pool.get_size())&&_data.get_size_1()==_res.get_size())
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

	template <typename T, bool initialize>
	tns<T, initialize> rotate(const tns<T, initialize>& _core)
	{
		if (_core.is_empty())
			return tns<T, initialize>();
		else
		{
			uint64_t size_1 = _core.get_size_1(), size_2 = _core.get_size_2(), size_3 = _core.get_size_3();
			tns<T, initialize> result(size_1, size_2, size_3);

			for (uint64_t i = (uint64_t)0, l = (uint64_t)0; i < size_1; ++i)
				for (uint64_t j = size_2 - (uint64_t)1;; --j)
				{
					for (uint64_t k = size_3 - (uint64_t)1;; --k)
					{
						result(i, j, k) = _core(l);

						++l;
						if (k == (uint64_t)0)
							break;
					}

					if (j == (uint64_t)0)
						break;
				}

			return result;
		}
	}
}