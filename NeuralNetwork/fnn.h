#pragma once

#include <cmath>
#include <cstdint>
#include <exception>
#include <random>
#include <utility>
#include <omp.h>

#include "rfm.h"
#include "nn_params.h"
#include "nn_activs.h"
#include "nn_derivs.h"
#include "nn_inits.h"
#include "light_appx.h"
#include "rand_sel.h"
#include "pipe.h"
#include "train_info.h"

using namespace std;

using namespace arithmetic;
using namespace nn_params;
using namespace nn_activs;
using namespace nn_derivs;
using namespace nn_inits;

template <nn_activ_t activ, nn_init_t init>
class fnnl
{
private:
	mtx<double> _weights;
	vec<double> _bias_ws;

public:
	fnnl() noexcept : _weights(), _bias_ws() {}

	fnnl(uint64_t isize, uint64_t osize, double param) : _weights(osize, isize), _bias_ws(osize)
	{
		if (isize == (uint64_t)0 || osize == (uint64_t)0)
			throw exception("FNN layer initialized with null");
		else
		{
			mt19937_64 rand_gen((random_device())());
			initializer<activ, init> dstr(isize, osize, param);

			for (uint64_t i = (uint64_t)0; i < isize * osize; ++i)
				_weights(i) = dstr(rand_gen);

			for (uint64_t i = (uint64_t)0; i < osize; ++i)
				_bias_ws(i) = (double)0;
		}
	}

	fnnl(const mtx<double>& weights, const vec<double>& bias_ws) : _weights(weights), _bias_ws(bias_ws) {}

	fnnl(mtx<double>&& weights, vec<double>&& bias_ws) noexcept : _weights(move(weights)), _bias_ws(move(bias_ws)) {}

	fnnl(const fnnl& o) : _weights(o._weights), _bias_ws(o._bias_ws) {}

	fnnl(fnnl&& o) noexcept : _weights(move(o._weights)), _bias_ws(move(o._bias_ws)) {}

	~fnnl() {}

	vec<double> pass_bwd(const vec<double>& output, double param) const
	{
		if (is_empty())
			throw exception("FNN layer is not initialized");
		else if (_weights.get_size_1() == output.get_size())
		{
			throw exception("Not implemented");
		}
		else
			throw exception("Given vector size and FNN layer output vector size do not match");
	}

	vec<double> pass_fwd(const vec<double>& input, double param) const
	{
		if (is_empty())
			throw exception("FNN layer is not initialized");
		else if (_weights.get_size_2() == input.get_size())
		{
			vec<double> result = _weights * input;

			__assume(result.get_size() == _bias_ws.get_size());
			for (uint64_t i = (uint64_t)0; i < result.get_size(); ++i)
				result(i) = activation<activ>(result(i) + _bias_ws(i), param);

			return result;
		}
		else
			throw exception("Given vector size and FNN layer input vector size do not match");
	}

	tuple<vec<double>, vec<double>> train_bwd(const vec<double>& grads, const vec<double>& grads_bias) const
	{
		if (is_empty())
			throw exception("FNN layer is not initialized");
		else if (_weights.get_size_1() == grads.get_size() && _weights.get_size_1() == grads_bias.get_size())
			return tuple<vec<double>, vec<double>>
			{
				move(grads* _weights),
				move(grads_bias* _weights)
			};
		else
			throw exception("Given vector size and FNN layer output vector size do not match");
	}

	tuple<vec<double>, vec<double>> train_fwd(const vec<double>& input, double param) const
	{
		if (is_empty())
			throw exception("FNN layer is not initialized");
		else if (_weights.get_size_2() == input.get_size())
		{
			tuple<vec<double>, vec<double>> result =
			{
				move(_weights * input),
				vec<double>(_weights.get_size_1())
			};

			__assume(get<(size_t)0>(result).get_size() == _weights.get_size_1());
			for (uint64_t i = (uint64_t)0; i < _weights.get_size_1(); ++i)
			{
				double x = get<(size_t)0>(result)(i) + _bias_ws(i);
				double y = activation<activ>(x, param);

				get<(size_t)0>(result)(i) = y;
				get<(size_t)1>(result)(i) = derivation<activ>(x, y, param);
			}

			return result;
		}
		else
			throw exception("Given vector size and FNN layer input vector size do not match");
	}

	void train_upd(const mtx<double>& weights_ds, const vec<double>& bias_ws_ds)
	{
		if (is_empty())
			throw exception("FNN layer is not initialized");
		else if (_weights.get_size_1() == weights_ds.get_size_1()
			&& _weights.get_size_2() == weights_ds.get_size_2()
			&& _bias_ws.get_size() == bias_ws_ds.get_size())
		{
			_weights += weights_ds;
			_bias_ws += bias_ws_ds;
		}
		else
			throw exception("Given deltas sizes and FNN weights sizes do not match");
	}

	uint64_t get_isize() const noexcept
	{
		return _weights.get_size_2();
	}

	uint64_t get_osize() const noexcept
	{
		return _weights.get_size_1();
	}

	uint64_t get_param_count() const noexcept
	{
		return _weights.get_size() + _bias_ws.get_size();
	}

	bool is_empty() const noexcept
	{
		return _weights.is_empty() || _bias_ws.is_empty();
	}

	fnnl& operator=(const fnnl& o)
	{
		_weights = o._weights;
		_bias_ws = o._bias_ws;

		return *this;
	}

	fnnl& operator=(fnnl&& o) noexcept
	{
		_weights = move(o._weights);
		_bias_ws = move(o._bias_ws);

		return *this;
	}
};

template <nn_activ_t activ, nn_init_t init = nn_init_t::normal>
class fnn
{
private:
	fnnl<activ, init>* _layers;
	uint64_t _count;
	double _param;

	bool check_set(const tuple<const vec<double>*,const vec<double>*,uint64_t>&set)
	{
		uint64_t last_layer=_count-(uint64_t)1;

		if(get<(size_t)0>(set)==nullptr
			||get<(size_t)1>(set)==nullptr
			||get<(size_t)2>(set)==(uint64_t)0)
			return false;

		for(uint64_t i=(uint64_t)0;i<get<(size_t)2>(set);++i)
			if(get<(size_t)0>(set)[i].is_empty()||_layers[(uint64_t)0].get_isize()!=get<(size_t)0>(set)[i].get_size()
				||get<(size_t)1>(set)[i].is_empty()||_layers[last_layer].get_osize()!=get<(size_t)1>(set)[i].get_size())
				return false;

		return true;
	}

	void split_set(tuple<const vec<double>*,const vec<double>*,uint64_t>&train_set,
		tuple<const vec<double>*,const vec<double>*,uint64_t>&test_set)
	{

	}

	void unite_set(tuple<const vec<double>*,const vec<double>*,uint64_t>&train_set,
		tuple<const vec<double>*,const vec<double>*,uint64_t>&test_set)
	{

	}

	bool use_cross_train_mode(tuple<const vec<double>*,const vec<double>*,uint64_t>&train_set,
		tuple<const vec<double>*,const vec<double>*,uint64_t>&test_set,bool _split_set)
	{
		bool need_ctm=get<(size_t)2>(train_set)+get<(size_t)2>(test_set)<get_param_count()*(uint64_t)25;

		if(check_set(test_set))
			if(need_ctm)
				return true;
			else
			{
				unite_set(train_set,test_set);
				return false;
			}
		else if(_split_set&&need_ctm)
		{
			split_set(train_set,test_set);
			return true;
		}
		else
			return false;
	}

public:
	fnn() noexcept : _layers(nullptr), _count((uint64_t)0), _param((double)1) {}

	fnn(uint64_t isize, uint64_t osize, uint64_t ssize, uint64_t count, double param = (double)1)
		: _count(count), _param(param)
	{
		if (isize == (uint64_t)0 || osize == (uint64_t)0
			|| (ssize == (uint64_t)0 && count > (uint64_t)1) || count == (uint64_t)0)
		{
			_layers = nullptr;
			_count = (uint64_t)0;
			_param = (double)1;

			throw exception("FNN layer(s) initialized with null(s)");
		}
		else
		{
			_layers = new fnnl<activ, init>[_count]();

			for (uint64_t i = (uint64_t)0; i < _count; ++i)
				_layers[i] = fnnl<activ, init>(
					i == (uint64_t)0 ? isize : ssize,
					i == _count - (uint64_t)1 ? osize : ssize, _param);
		}
	}

	fnn(const fnn_layer_info* info, uint64_t count, double param = (double)1)
		: _count(count), _param(param)
	{
		if (info == nullptr || count == (uint64_t)0)
		{
			_layers = nullptr;
			_count = (uint64_t)0;
			_param = (double)1;

			throw exception("FNN layer(s) initialized with null(s)");
		}
		else
		{
			_layers = new fnnl<activ, init>[_count]();

			for (uint64_t i = (uint64_t)0; i < _count; ++i)
				if (i == (uint64_t)0 || info[i].isize == info[i - (uint64_t)1].osize)
					_layers[i] = fnnl<activ, init>(info[i].isize, info[i].osize, _param);
				else
				{
					if (_layers != nullptr)
						delete[] _layers;

					_layers = nullptr;
					_count = (uint64_t)0;
					_param = (double)1;

					throw exception("FNN layers sizes do not match");
				}
		}
	}

	fnn(initializer_list<uint64_t> sizes, double param = (double)1)
		: _count(sizes.size() <= (uint64_t)1 ? (uint64_t)0 : sizes.size() - (uint64_t)1), _param(param)
	{
		if (_count == (uint64_t)0)
		{
			_layers = nullptr;
			_count = (uint64_t)0;
			_param = (double)1;

			throw exception("FNN layer(s) initialized with null(s)");
		}
		else
		{
			_layers = new fnnl<activ, init>[_count]();

			const uint64_t* sizes_arr = sizes.begin();
			for (uint64_t i = (uint64_t)0; i < _count; ++i)
				_layers[i] = fnnl<activ, init>(sizes_arr[i], sizes_arr[i + (uint64_t)1], _param);
		}
	}

	fnn(const fnn& o) : _count(o._count), _param(o._param)
	{
		if (o.is_empty())
		{
			_layers = nullptr;
			_count = (uint64_t)0;
			_param = (double)1;
		}
		else
		{
			_layers = new fnnl<activ, init>[_count]();

			for (uint64_t i = (uint64_t)0; i < _count; ++i)
				_layers[i] = o._layers[i];
		}
	}

	fnn(fnn&& o) noexcept : _layers(o._layers), _count(o._count), _param(o._param)
	{
		o._layers = nullptr;
		o._count = (uint64_t)0;
		o._param = (double)1;
	}

	~fnn()
	{
		if (_layers != nullptr)
			delete[] _layers;
	}

	vec<double> pass_bwd(const vec<double>& output) const
	{
		if (is_empty())
			throw exception("FNN is not initialized");
		else if (_layers[_count - (uint64_t)1].get_osize() == output.get_size())
		{
			throw exception("Not implemented");
		}
		else
			throw exception("Given vector size and FNN output vector size do not match");
	}

	vec<double> pass_bwd(vec<double>&& output) const
	{
		if (is_empty())
			throw exception("FNN is not initialized");
		else if (_layers[_count - (uint64_t)1].get_osize() == output.get_size())
		{
			throw exception("Not implemented");
		}
		else
			throw exception("Given vector size and FNN output vector size do not match");
	}

	vec<double> pass_fwd(const vec<double>& input) const
	{
		if (is_empty())
			throw exception("FNN is not initialized");
		else if (_layers[(uint64_t)0].get_isize() == input.get_size())
		{
			vec<double> result = _layers[(uint64_t)0].pass_fwd(input, _param);

			for (uint64_t i = (uint64_t)1; i < _count; ++i)
				result = _layers[i].pass_fwd(result, _param);

			return result;
		}
		else
			throw exception("Given vector size and FNN input vector size do not match");
	}

	vec<double> pass_fwd(vec<double>&& input) const
	{
		if (is_empty())
			throw exception("FNN is not initialized");
		else if (_layers[(uint64_t)0].get_isize() == input.get_size())
		{
			for (uint64_t i = (uint64_t)0; i < _count; ++i)
				input = _layers[i].pass_fwd(input, _param);

			return input;
		}
		else
			throw exception("Given vector size and FNN input vector size do not match");
	}

	tuple<vec<double>, vec<double>> train_batch_mode(tuple<const vec<double>*, const vec<double>*, uint64_t> train_set,
		tuple<const vec<double>*, const vec<double>*, uint64_t> test_set = { nullptr, nullptr, (uint64_t)0 },
		pipe<tuple<double, double>>* error_pipe = nullptr, uint64_t max_epochs = (uint64_t)10000,
		uint64_t test_freq = (uint64_t)100, bool _split_set = false, double convergence = (double)1,
		double max_error = (double)0.1, double min_error = (double)1e-10,
		double alpha_start = (double)0.9, double alpha_end = (double)0,
		double speed_start = (double)0.9, double speed_end = (double)0.01)
	{
		if (is_empty())
			throw exception("FNN is not initialized");
		else if (check_set(train_set))
		{
			bool cross_train_mode = use_cross_train_mode(train_set, test_set, _split_set);

			vec<double> train_errs(max_epochs);
			vec<double> test_errs(max_epochs);
			test_errs((uint64_t)0) = (double)0;

			double min_test_err;
			convergence *= (double)_count;
			uint64_t last_layer = _count - (uint64_t)1;

			uint64_t copy_count = (uint64_t)(omp_get_max_threads() * (uint64_t)2 < get<(size_t)2>(train_set)
				? omp_get_max_threads() * (uint64_t)2 : get<(size_t)2>(train_set));
			uint64_t cycles = get<(size_t)2>(train_set) / copy_count
				+ (get<(size_t)2>(train_set) % copy_count == (uint64_t)0 ? (uint64_t)0 : (uint64_t)1);

			light_appx alpha_appxer(alpha_start, alpha_end, max_epochs);
			light_appx speed_appxer(speed_start, speed_end, max_epochs);

			rand_sel_i<uint64_t>* selector = max_epochs > (uint64_t)25
				? (rand_sel_i<uint64_t>*)(new rand_sel<uint64_t, false>(get<(size_t)2>(train_set) - (uint64_t)1, (uint64_t)0))
				: (rand_sel_i<uint64_t>*)(new rand_sel<uint64_t, true>(get<(size_t)2>(train_set) - (uint64_t)1, (uint64_t)0));
			uint64_t* selectors = new uint64_t[copy_count];

			vec<double>** arr_activs = new vec<double>*[copy_count];
			vec<double>** arr_derivs = new vec<double>*[copy_count];

			vec<double>** arr_grads = new vec<double>*[copy_count];
			vec<double>** arr_grads_bias = new vec<double>*[copy_count];

			mtx<double>* deltas = new mtx<double>[_count]();
			vec<double>* deltas_bias = new vec<double>[_count]();

			for (uint64_t i = (uint64_t)0; i < copy_count; ++i)
			{
				arr_activs[i] = new vec<double>[_count]();
				arr_derivs[i] = new vec<double>[_count]();
				arr_grads[i] = new vec<double>[_count]();
				arr_grads_bias[i] = new vec<double>[_count]();

				arr_grads[i][last_layer] = vec<double>(_layers[last_layer].get_osize());
				arr_grads_bias[i][last_layer] = vec<double>(_layers[last_layer].get_osize());
			}

			for (uint64_t i = (uint64_t)0; i < _count; ++i)
			{
				deltas[i] = mtx<double>(_layers[i].get_osize(), _layers[i].get_isize());
				deltas_bias[i] = vec<double>(_layers[i].get_osize());

				for (uint64_t j = (uint64_t)0; j < deltas[i].get_size(); ++j)
					deltas[i](j) = (double)0;

				for (uint64_t j = (uint64_t)0; j < deltas_bias[i].get_size(); ++j)
					deltas_bias[i](j) = (double)0;
			}

			for (uint64_t i = (uint64_t)0; i < max_epochs; ++i)
			{
				double train_err = (double)0, test_err = (double)0;
				double alpha_epo = alpha_appxer.forward(), speed_epo = speed_appxer.forward();

				light_appx conv_appxer(max(convergence, (double)1) * speed_epo, speed_epo, _count, (double)0, (double)2);
				selector->reset();

				for (uint64_t j = (uint64_t)0; j < cycles; ++j)
				{
					uint64_t copy_left = j == cycles - (uint64_t)1 ? get<(size_t)2>(train_set) - j * copy_count : copy_count;

					for (uint64_t copy_num = (uint64_t)0; copy_num < copy_left; ++copy_num)
						selectors[copy_num] = selector->next();

#pragma omp parallel for reduction(+:train_err) num_threads(copy_left)
					for (int64_t omp_thread = (int64_t)0; omp_thread < copy_left; ++omp_thread)
					{
						uint64_t num = selectors[omp_thread];

						vec<double>* activs = arr_activs[omp_thread];
						vec<double>* derivs = arr_derivs[omp_thread];

						vec<double>* grads = arr_grads[omp_thread];
						vec<double>* grads_bias = arr_grads_bias[omp_thread];

						for (uint64_t k = (uint64_t)0; k < _count; ++k)
						{
							uint64_t k_prev = k - (uint64_t)1;
							tuple<vec<double>, vec<double>> train_fwd_result =
								_layers[k].train_fwd(k == (uint64_t)0 ? get<(size_t)0>(train_set)[num] : activs[k - (uint64_t)1], _param);

							activs[k] = move(get<(size_t)0>(train_fwd_result));
							derivs[k] = move(get<(size_t)1>(train_fwd_result));
						}

						__assume(last_layer < _count);
						for (uint64_t k = (uint64_t)0; k < _layers[last_layer].get_osize(); ++k)
						{
							double loss = get<(size_t)1>(train_set)[num](k) - activs[last_layer](k);

							grads[last_layer](k) = derivs[last_layer](k) * loss;
							grads_bias[last_layer](k) = loss;
							train_err += loss * loss;
						}

						__assume(last_layer < _count);
						for (uint64_t k = last_layer; k > (uint64_t)0; --k)
						{
							__assume(k < _count);
							uint64_t k_prev = k - (uint64_t)1;
							tuple<vec<double>, vec<double>> train_bwd_result =
								_layers[k].train_bwd(grads[k], grads_bias[k]);

							__assume(k_prev >= (uint64_t)0);
							grads[k_prev] = move(get<(size_t)0>(train_bwd_result));
							grads_bias[k_prev] = move(get<(size_t)1>(train_bwd_result));

							for (uint64_t l = (uint64_t)0; l < grads[k_prev].get_size(); ++l)
								grads[k_prev](l) *= derivs[k_prev](l);
						}
					}

					conv_appxer.reset();

					for (uint64_t k = (uint64_t)0; k < _count; ++k)
					{
						double speed_lay = conv_appxer.forward() / (double)copy_left;

						deltas[k] *= alpha_epo;
						deltas_bias[k] *= alpha_epo;

						for (uint64_t copy_num = (uint64_t)0; copy_num < copy_left; ++copy_num)
						{
							uint64_t num = selectors[copy_num];
							uint64_t l = (uint64_t)0;

							__assume(deltas[k].get_size_1() == deltas_bias[k].get_size());
							for (uint64_t m = (uint64_t)0; m < deltas[k].get_size_1(); ++m)
							{
								for (uint64_t n = (uint64_t)0; n < deltas[k].get_size_2(); ++n, ++l)
									deltas[k](l) += arr_grads[copy_num][k](m) * (k == (uint64_t)0
										? get<(size_t)0>(train_set)[num] : arr_activs[copy_num][k - (uint64_t)1])(n) * speed_lay;

								deltas_bias[k](m) += arr_grads_bias[copy_num][k](m) * speed_lay;
							}
						}

						_layers[k].train_upd(deltas[k], deltas_bias[k]);
					}
				}

				if (cross_train_mode && i % test_freq == (uint64_t)0)
				{
#pragma omp parallel for reduction(+:test_err) num_threads(copy_count)
					for (uint64_t j = (uint64_t)0; j < get<(size_t)2>(test_set); ++j)
					{
						vec<double> fwd_result = pass_fwd(get<(size_t)0>(test_set)[j]);

						for (uint64_t k = (uint64_t)0; k < fwd_result.get_size(); ++k)
						{
							double loss = get<(size_t)1>(test_set)[j](k) - fwd_result(k);
							test_err += loss * loss;
						}
					}

					test_err /= (double)get<(size_t)2>(test_set);
				}
				else
					test_err = test_errs[i - (uint64_t)1];

				train_err /= (double)get<(size_t)2>(train_set);

				if (error_pipe != nullptr)
					error_pipe->push(move(tuple<double, double>{ train_err, test_err }));

				if (test_err < min_test_err)
					min_test_err = test_err;

				if (test_err > min_test_err * max_error || train_err < min_error)
				{
					vec<double> new_train_errs(i + (uint64_t)1);
					vec<double> new_test_errs(i + (uint64_t)1);

					for (uint64_t j = (uint64_t)0; j < i; ++j)
					{
						new_train_errs(j) = train_errs(j);
						new_test_errs(j) = test_errs(j);
					}

					new_train_errs(i) = train_err;
					new_test_errs(i) = test_err;

					train_errs = move(new_train_errs);
					test_errs = move(new_test_errs);

					break;
				}
				else
				{
					train_errs(i) = train_err;
					test_errs(i) = test_err;
				}
			}

			for (uint64_t i = (uint64_t)0; i < copy_count; ++i)
			{
				delete[] arr_activs[i];
				delete[] arr_derivs[i];
				delete[] arr_grads[i];
				delete[] arr_grads_bias[i];
			}

			delete selector;
			delete[] selectors;

			delete[] arr_activs;
			delete[] arr_derivs;

			delete[] arr_grads;
			delete[] arr_grads_bias;

			delete[] deltas;
			delete[] deltas_bias;

			return tuple<vec<double>, vec<double>>
			{
				train_errs,
					test_errs
			};
		}
		else
			throw exception("Wrong data set");
	}

	tuple<vec<double>, vec<double>> train_stoch_mode(tuple<const vec<double>*, const vec<double>*, uint64_t> train_set,
		tuple<const vec<double>*, const vec<double>*, uint64_t> test_set = { nullptr, nullptr, (uint64_t)0 },
		pipe<tuple<double, double>>* error_pipe = nullptr, uint64_t max_epochs = (uint64_t)10000,
		uint64_t test_freq = (uint64_t)100, double convergence = (double)1, bool _split_set = false,
		double max_error = (double)0.1, double min_error = (double)1e-10,
		double alpha_start = (double)0.9, double alpha_end = (double)0,
		double speed_start = (double)0.9, double speed_end = (double)0.01)
	{
		if (is_empty())
			throw exception("FNN is not initialized");
		else if (check_set(train_set))
		{
			bool cross_train_mode = use_cross_train_mode(train_set, test_set, _split_set);

			vec<double> train_errs(max_epochs);
			vec<double> test_errs(max_epochs);
			test_errs((uint64_t)0) = (double)0;

			double min_test_err;
			convergence *= (double)_count;
			uint64_t last_layer = _count - (uint64_t)1;

			light_appx alpha_appxer(alpha_start, alpha_end, max_epochs);
			light_appx speed_appxer(speed_start, speed_end, max_epochs);

			rand_sel_i<uint64_t>* selector = max_epochs > (uint64_t)25
				? (rand_sel_i<uint64_t>*)(new rand_sel<uint64_t, false>(get<(size_t)2>(train_set) - (uint64_t)1, (uint64_t)0))
				: (rand_sel_i<uint64_t>*)(new rand_sel<uint64_t, true>(get<(size_t)2>(train_set) - (uint64_t)1, (uint64_t)0));

			vec<double>* activs = new vec<double>[_count]();
			vec<double>* derivs = new vec<double>[_count]();

			vec<double>* grads = new vec<double>[_count]();
			vec<double>* grads_bias = new vec<double>[_count]();

			mtx<double>* deltas = new mtx<double>[_count]();
			vec<double>* deltas_bias = new vec<double>[_count]();

			grads[last_layer] = vec<double>(_layers[last_layer].get_osize());
			grads_bias[last_layer] = vec<double>(_layers[last_layer].get_osize());

			for (uint64_t i = (uint64_t)0; i < _count; ++i)
			{
				uint64_t mtx_size = deltas[i].get_size_1() * deltas[i].get_size_2();
				uint64_t vec_size = deltas_bias[i].get_size();

				for (uint64_t j = (uint64_t)0; j < deltas[i].get_size(); ++j)
					deltas[i](j) = (double)0;

				for (uint64_t j = (uint64_t)0; j < deltas_bias[i].get_size(); ++j)
					deltas_bias[i](j) = (double)0;
			}

			for (uint64_t i = (uint64_t)0; i < max_epochs; ++i)
			{
				double train_err = (double)0, test_err = (double)0;
				double alpha_epo = alpha_appxer.forward(), speed_epo = speed_appxer.forward();

				light_appx conv_appxer(max(convergence, (double)1) * speed_epo, speed_epo, _count, (double)0, (double)2);
				selector->reset();

				for (int64_t j = (int64_t)0; j < get<(size_t)2>(train_set); ++j)
				{
					uint64_t num = selector->next();

					for (uint64_t k = (uint64_t)0; k < _count; ++k)
					{
						uint64_t k_prev = k - (uint64_t)1;
						tuple<vec<double>, vec<double>> train_fwd_result =
							_layers[k].train_fwd(k == (uint64_t)0 ? get<(size_t)0>(train_set)[num] : activs[k - (uint64_t)1], _param);

						activs[k] = move(get<(size_t)0>(train_fwd_result));
						derivs[k] = move(get<(size_t)1>(train_fwd_result));
					}

					__assume(last_layer < _count);
					for (uint64_t k = (uint64_t)0; k < _layers[last_layer].get_osize(); ++k)
					{
						double loss = get<(size_t)1>(train_set)[num](k) - activs[last_layer](k);

						grads[last_layer](k) = derivs[last_layer](k) * loss;
						grads_bias[last_layer](k) = loss;
						train_err += loss * loss;
					}

					__assume(last_layer < _count);
					for (uint64_t k = last_layer; k > (uint64_t)0; --k)
					{
						__assume(k < _count);
						uint64_t k_prev = k - (uint64_t)1;
						tuple<vec<double>, vec<double>> train_bwd_result =
							_layers[k].train_bwd(grads[k], grads_bias[k]);

						__assume(k_prev >= (uint64_t)0);
						grads[k_prev] = move(get<(size_t)0>(train_bwd_result));
						grads_bias[k_prev] = move(get<(size_t)1>(train_bwd_result));

						for (uint64_t l = (uint64_t)0; l < grads[k_prev].get_size(); ++l)
							grads[k_prev](l) *= derivs[k_prev](l);
					}

					conv_appxer.reset();

					for (uint64_t k = (uint64_t)0; k < _count; ++k)
					{
						double speed_lay = conv_appxer.forward();
						uint64_t l = (uint64_t)0;

						__assume(deltas[k].get_size_1() == deltas_bias[k].get_size());
						for (uint64_t m = (uint64_t)0; m < deltas[k].get_size_1(); ++m)
						{
							for (uint64_t n = (uint64_t)0; n < deltas[k].get_size_2(); ++n, ++l)
								deltas[k](l) = deltas[k](l) * alpha_epo + grads[k](m) * (k == (uint64_t)0
									? get<(size_t)0>(train_set)[num] : activs[k - (uint64_t)1])(n) * speed_lay;

							deltas_bias[k](m) = deltas_bias[k](m) * alpha_epo + grads_bias[k](m) * speed_lay;
						}

						_layers[k].train_upd(deltas[k], deltas_bias[k]);
					}
				}

				if (cross_train_mode && i % test_freq == (uint64_t)0)
				{
					for (uint64_t j = (uint64_t)0; j < get<(size_t)2>(test_set); ++j)
					{
						vec<double> fwd_result = pass_fwd(get<(size_t)0>(test_set)[j]);

						for (uint64_t k = (uint64_t)0; k < fwd_result.get_size(); ++k)
						{
							double loss = get<(size_t)1>(test_set)[j](k) - fwd_result(k);
							test_err += loss * loss;
						}
					}

					test_err /= (double)get<(size_t)2>(test_set);
				}
				else
					test_err = test_errs[i - (uint64_t)1];

				train_err /= (double)get<(size_t)2>(train_set);

				if (error_pipe != nullptr)
					error_pipe->push(move(tuple<double, double>{ train_err, test_err }));

				if (test_err < min_test_err)
					min_test_err = test_err;

				if (test_err > min_test_err * max_error || train_err < min_error)
				{
					vec<double> new_train_errs(i + (uint64_t)1);
					vec<double> new_test_errs(i + (uint64_t)1);

					for (uint64_t j = (uint64_t)0; j < i; ++j)
					{
						new_train_errs(j) = train_errs(j);
						new_test_errs(j) = test_errs(j);
					}

					new_train_errs(i) = train_err;
					new_test_errs(i) = test_err;

					train_errs = move(new_train_errs);
					test_errs = move(new_test_errs);

					break;
				}
				else
				{
					train_errs(i) = train_err;
					test_errs(i) = test_err;
				}
			}

			delete selector;

			delete[] activs;
			delete[] derivs;

			delete[] grads;
			delete[] grads_bias;

			delete[] deltas;
			delete[] deltas_bias;

			return tuple<vec<double>, vec<double>>
			{
				train_errs,
					test_errs
			};
		}
		else
			throw exception("Wrong data set");
	}

	uint64_t get_count() const noexcept
	{
		return _count;
	}

	double get_param() const noexcept
	{
		return _param;
	}

	uint64_t get_param_count() const noexcept
	{
		uint64_t result=(uint64_t)0;

		for (uint64_t i=(uint64_t)0;i<_count;++i)
			result+=_layers[i].get_param_count();

		return result;
	}

	bool is_empty() const noexcept
	{
		return _layers==nullptr;
	}

	fnn& operator=(const fnn& o)
	{
		if (_layers != nullptr)
			delete[] _layers;

		if (o.is_empty())
		{
			_layers = nullptr;
			_count = (uint64_t)0;
			_param = (double)1;
		}
		else
		{
			_layers = new fnnl<activ, init>[o._count]();
			_count = o._count;
			_param = o._param;

			for (uint64_t i = (uint64_t)0; i < _count; ++i)
				_layers[i] = o._layers[i];
		}

		return *this;
	}

	fnn& operator=(fnn&& o) noexcept
	{
		if (_layers != nullptr)
			delete[] _layers;

		_layers = o._layers;
		_count = o._count;
		_param = o._param;

		o._layers = nullptr;
		o._count = (uint64_t)0;
		o._param = (double)1;

		return *this;
	}
};