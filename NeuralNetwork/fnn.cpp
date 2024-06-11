#include "fnn.h"

#include <cmath>
#include <exception>
#include <random>
#include <utility>
#include <omp.h>

#include "light_appx.h"
#include "rand_sel.h"

using namespace std;
using namespace arithmetic;
using namespace nn_params;

fnnl::fnnl() noexcept : _weights(), _bias_ws(), _activ(nn_activ_t::__count__), _param((double)1) {}

fnnl::fnnl(uint64_t isize, uint64_t osize, nn_activ_t activ, nn_init_t init, double param)
	: _weights(osize, isize), _bias_ws(osize), _activ(activ), _param(param)
{
	if (isize == (uint64_t)0 || osize == (uint64_t)0 || param == (double)0
		|| activ >= nn_activ_t::__count__ || init >= nn_init_t::__count__)
		throw exception("FNN layer initialized with null");
	else
	{
		rand_init_base* randomizer = nullptr;

		__assume(init < nn_init_t::__count__);
		switch (init)
		{
		case nn_init_t::normal:
			randomizer = new rand_init<nn_init_t::normal>(isize, osize, param, activ);
			break;

		case nn_init_t::uniform:
			randomizer = new rand_init<nn_init_t::uniform>(isize, osize, param, activ);
			break;

		default:
			__assume(0);
		}

		for (uint64_t i = (uint64_t)0; i < _weights.get_size(); ++i)
			_weights(i) = (*randomizer)();

		for (uint64_t i = (uint64_t)0; i < _bias_ws.get_size(); ++i)
			_bias_ws(i) = (double)0;

		delete randomizer;
	}
}

fnnl::fnnl(const mtx<double>& weights, const vec<double>& bias_ws, nn_activ_t activ, double param)
	: _weights(weights), _bias_ws(bias_ws), _activ(activ), _param(param) {}

fnnl::fnnl(mtx<double>&& weights, vec<double>&& bias_ws, nn_activ_t activ, double param) noexcept
	: _weights(move(weights)), _bias_ws(move(bias_ws)), _activ(activ), _param(param) {}

fnnl::fnnl(const fnnl& o) : _weights(o._weights), _bias_ws(o._bias_ws), _activ(o._activ), _param(o._param) {}

fnnl::fnnl(fnnl&& o) noexcept : _weights(move(o._weights)), _bias_ws(move(o._bias_ws)), _activ(o._activ), _param(o._param) {}

fnnl::~fnnl() {}

vec<double> fnnl::pass_bwd(const vec<double>& output) const
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

vec<double> fnnl::pass_fwd(const vec<double>& input) const
{
	if (is_empty())
		throw exception("FNN layer is not initialized");
	else if (_weights.get_size_2() == input.get_size())
	{
		vec<double> result(move(_weights * input));

		__assume(result.get_size() == _bias_ws.get_size());
		for (uint64_t i = (uint64_t)0; i < result.get_size(); ++i)
			result(i) = activation(result(i) + _bias_ws(i), _param, _activ);

		return result;
	}
	else
		throw exception("Given vector size and FNN layer input vector size do not match");
}

tuple<vec<double>, vec<double>> fnnl::train_bwd(const vec<double>& grads, const vec<double>& grads_bias) const
{
	if (is_empty())
		throw exception("FNN layer is not initialized");
	else if (_weights.get_size_1() == grads.get_size()
		&& _weights.get_size_1() == grads_bias.get_size())
		return
	{
		move(grads * _weights),
		move(grads_bias * _weights)
	};
	else
		throw exception("Given vector size and FNN layer output vector size do not match");
}

tuple<vec<double>, vec<double>> fnnl::train_fwd(const vec<double>& input) const
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
		__assume(get<(size_t)1>(result).get_size() == _weights.get_size_1());
		for (uint64_t i = (uint64_t)0; i < _weights.get_size_1(); ++i)
		{
			double x = get<(size_t)0>(result)(i) + _bias_ws(i);
			double y = activation(x, _param, _activ);

			get<(size_t)0>(result)(i) = y;
			get<(size_t)1>(result)(i) = derivation(x, y, _param, _activ);
		}

		return result;
	}
	else
		throw exception("Given vector size and FNN layer input vector size do not match");
}

void fnnl::train_upd(const mtx<double>& weights_ds, const vec<double>& bias_ws_ds)
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

inline nn_activ_t fnnl::get_activ_type() const noexcept
{
	return _activ;
}

inline uint64_t fnnl::get_isize() const noexcept
{
	return _weights.get_size_2();
}

inline uint64_t fnnl::get_osize() const noexcept
{
	return _weights.get_size_1();
}

inline double fnnl::get_param() const noexcept
{
	return _param;
}

inline uint64_t fnnl::get_param_count() const noexcept
{
	return _weights.get_size() + _bias_ws.get_size();
}

inline bool fnnl::is_empty() const noexcept
{
	return _weights.is_empty() || _bias_ws.is_empty();
}

fnnl& fnnl::operator=(const fnnl& o)
{
	_weights = o._weights;
	_bias_ws = o._bias_ws;
	_activ = o._activ;
	_param = o._param;

	return *this;
}

fnnl& fnnl::operator=(fnnl&& o) noexcept
{
	_weights = move(o._weights);
	_bias_ws = move(o._bias_ws);
	_activ = o._activ;
	_param = o._param;

	return *this;
}

inline void fnn::add_info(list<info>* data, pipe<info>* data_pipe, const info& _info)
{
	if (data != nullptr)
		data->push_back(_info);

	if (data_pipe != nullptr)
		data_pipe->push(_info);
}

bool fnn::check_set(const tuple<const vec<double>*, const vec<double>*, uint64_t>& set)
{
	uint64_t isize = _layers[(uint64_t)0].get_isize(), osize = _layers[_count - (uint64_t)1].get_osize();

	if (get<(size_t)0>(set) == nullptr || get<(size_t)1>(set) == nullptr || get<(size_t)2>(set) == (uint64_t)0)
		return false;

	for (uint64_t i = (uint64_t)0; i < get<(size_t)2>(set); ++i)
		if (get<(size_t)0>(set)[i].is_empty() || get<(size_t)0>(set)[i].get_size() != isize
			|| get<(size_t)1>(set)[i].is_empty() || get<(size_t)1>(set)[i].get_size() != osize)
			return false;

	return true;
}

void fnn::split_set(tuple<const vec<double>*, const vec<double>*, uint64_t>& src_set,
	tuple<vec<double>*, vec<double>*, uint64_t>& train_set,
	tuple<vec<double>*, vec<double>*, uint64_t>& test_set)
{
	uint64_t p_count = max(get_param_count(), (uint64_t)2);

	get<(size_t)2>(train_set) = (uint64_t)trunc((double)get<(size_t)2>(src_set)
		* ((double)1 - (sqrt((double)(p_count * (uint64_t)2 - (uint64_t)1)) - (double)1)
			/ (double)((p_count - (uint64_t)1) * (uint64_t)2)));
	get<(size_t)2>(test_set) = get<(size_t)2>(src_set) - get<(size_t)2>(train_set);

	get<(size_t)0>(train_set) = new vec<double>[get<(size_t)2>(train_set)]();
	get<(size_t)1>(train_set) = new vec<double>[get<(size_t)2>(train_set)]();

	get<(size_t)0>(test_set) = new vec<double>[get<(size_t)2>(test_set)]();
	get<(size_t)1>(test_set) = new vec<double>[get<(size_t)2>(test_set)]();

	rand_sel<uint64_t, true> selector(get<(size_t)2>(src_set) - (uint64_t)1, (uint64_t)0);

	for (uint64_t i = (uint64_t)0; i < get<(size_t)2>(train_set); ++i)
	{
		uint64_t num = selector.next();

		get<(size_t)0>(train_set)[i] = get<(size_t)0>(src_set)[num];
		get<(size_t)1>(train_set)[i] = get<(size_t)1>(src_set)[num];
	}

	for (uint64_t i = (uint64_t)0; i < get<(size_t)2>(test_set); ++i)
	{
		uint64_t num = selector.next();

		get<(size_t)0>(test_set)[i] = get<(size_t)0>(src_set)[num];
		get<(size_t)1>(test_set)[i] = get<(size_t)1>(src_set)[num];
	}
}

void fnn::unite_set(tuple<vec<double>*, vec<double>*, uint64_t>& dst_set,
	tuple<const vec<double>*, const vec<double>*, uint64_t>& train_set,
	tuple<const vec<double>*, const vec<double>*, uint64_t>& test_set)
{
	get<(size_t)2>(dst_set) = get<(size_t)2>(train_set) + get<(size_t)2>(test_set);

	get<(size_t)0>(dst_set) = new vec<double>[get<(size_t)2>(dst_set)]();
	get<(size_t)1>(dst_set) = new vec<double>[get<(size_t)2>(dst_set)]();

	for (uint64_t i = (uint64_t)0; i < get<(size_t)2>(train_set); ++i)
	{
		get<(size_t)0>(dst_set)[i] = get<(size_t)0>(train_set)[i];
		get<(size_t)1>(dst_set)[i] = get<(size_t)1>(train_set)[i];
	}

	for (uint64_t i = (uint64_t)0, j = get<(size_t)2>(train_set); i < get<(size_t)2>(test_set); ++i, ++j)
	{
		get<(size_t)0>(dst_set)[j] = get<(size_t)0>(test_set)[i];
		get<(size_t)1>(dst_set)[j] = get<(size_t)1>(test_set)[i];
	}
}

bool fnn::use_cross_train_mode(tuple<const vec<double>*, const vec<double>*, uint64_t>& train_set,
	tuple<const vec<double>*, const vec<double>*, uint64_t>& test_set, bool _split_set,
	tuple<vec<double>*, vec<double>*, uint64_t>& train_set_cross,
	tuple<vec<double>*, vec<double>*, uint64_t>& test_set_cross)
{
	bool need_ctm = get<(size_t)2>(train_set) + get<(size_t)2>(test_set) < get_param_count() * (uint64_t)50;

	if (check_set(test_set))
		if (need_ctm)
			return true;
		else
		{
			unite_set(train_set_cross, train_set, test_set);
			return false;
		}
	else if (_split_set && need_ctm)
	{
		split_set(train_set, train_set_cross, test_set_cross);
		return true;
	}
	else
		return false;
}

fnn::fnn() noexcept : _layers(nullptr), _count((uint64_t)0) {}

fnn::fnn(uint64_t isize, uint64_t osize, uint64_t ssize, uint64_t count, nn_activ_t activ,nn_init_t init, double param)
	: _count(count)
{
	if (isize == (uint64_t)0 || osize == (uint64_t)0 || count == (uint64_t)0|| (ssize == (uint64_t)0 && count > (uint64_t)1))
	{
		_layers = nullptr;
		_count = (uint64_t)0;

		throw exception(error_msg::fnn_wrong_initialize);
	}
	else
	{
		_layers = new fnnl[_count]();

		for (uint64_t i = (uint64_t)0; i < _count; ++i)
			_layers[i] = fnnl(i == (uint64_t)0 ? isize : ssize, i == _count - (uint64_t)1 ? osize : ssize,
				activ, init, param);
	}
}

fnn::fnn(initializer_list<uint64_t> sizes, nn_activ_t activ, nn_init_t init, double param)
	: _count(sizes.size() <= (uint64_t)1 ? (uint64_t)0 : sizes.size() - (uint64_t)1)
{
	if (_count == (uint64_t)0 || param == (double)0 || activ >= nn_activ_t::__count__ || init >= nn_init_t::__count__)
	{
		_layers = nullptr;
		_count = (uint64_t)0;

		throw exception(error_msg::fnn_wrong_initialize);
	}
	else
	{
		_layers = new fnnl[_count]();
		const uint64_t* sizes_arr = sizes.begin();

		for (uint64_t i = (uint64_t)0; i < _count; ++i)
			_layers[i] = fnnl(sizes_arr[i], sizes_arr[i + (uint64_t)1], activ, init, param);
	}
}

fnn::fnn(const fnn_layer_info* info, uint64_t count) : _count(count)
{
	if (info == nullptr || count == (uint64_t)0)
	{
		_layers = nullptr;
		_count = (uint64_t)0;

		throw exception(error_msg::fnn_wrong_initialize);
	}
	else
	{
		_layers = new fnnl[_count]();

		for (uint64_t i = (uint64_t)0; i < _count; ++i)
			if ((i == (uint64_t)0 ? info[i].isize == (uint64_t)0 : info[i].isize != info[i - (uint64_t)1].osize)
				|| info[i].osize == (uint64_t)0 || info[i].param == (double)0
				|| info[i].activ >= nn_activ_t::__count__ || info[i].init >= nn_init_t::__count__)
			{
				delete[] _layers;

				_layers = nullptr;
				_count = (uint64_t)0;

				throw exception(error_msg::fnn_wrong_initialize);
			}
			else
				_layers[i] = fnnl(info[i].isize, info[i].osize, info[i].activ, info[i].init, info[i].param);
	}
}

fnn::fnn(const fnn& o) : _count(o._count)
{
	if (o.is_empty())
	{
		_layers = nullptr;
		_count = (uint64_t)0;
	}
	else
	{
		_layers = new fnnl[_count]();

		for (uint64_t i = (uint64_t)0; i < _count; ++i)
			_layers[i] = o._layers[i];
	}
}

fnn::fnn(fnn&& o) noexcept : _layers(o._layers), _count(o._count)
{
	o._layers = nullptr;
	o._count = (uint64_t)0;
}

fnn::~fnn()
{
	if (_layers != nullptr)
		delete[] _layers;
}

vec<double> fnn::pass_bwd(const vec<double>& output) const
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

vec<double> fnn::pass_bwd(vec<double>&& output) const
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

vec<double> fnn::pass_fwd(const vec<double>& input) const
{
	if (is_empty())
		throw exception("FNN is not initialized");
	else if (_layers[(uint64_t)0].get_isize() == input.get_size())
	{
		vec<double> result(move(_layers[(uint64_t)0].pass_fwd(input)));

		for (uint64_t i = (uint64_t)1; i < _count; ++i)
			result = move(_layers[i].pass_fwd(result));

		return result;
	}
	else
		throw exception("Given vector size and FNN input vector size do not match");
}

vec<double> fnn::pass_fwd(vec<double>&& input) const
{
	if (is_empty())
		throw exception("FNN is not initialized");
	else if (_layers[(uint64_t)0].get_isize() == input.get_size())
	{
		for (uint64_t i = (uint64_t)0; i < _count; ++i)
			input = move(_layers[i].pass_fwd(input));

		return input;
	}
	else
		throw exception("Given vector size and FNN input vector size do not match");
}

void fnn::train_batch_mode
(
	tuple<const vec<double>*, const vec<double>*, uint64_t> train_set,
	tuple<const vec<double>*, const vec<double>*, uint64_t> test_set,
	list<info>* errors,
	pipe<info>* error_pipe,
	uint64_t max_epochs,
	uint64_t max_overs,
	uint64_t test_freq,
	bool _split_set,
	double convergence,
	double alpha_start,
	double alpha_end,
	double speed_start,
	double speed_end,
	double max_error,
	double min_error
)
{
	if (is_empty())
		throw exception("FNN is not initialized");
	else if (check_set(train_set))
	{
		tuple<vec<double>*, vec<double>*, uint64_t> train_set_cross =
		{
			nullptr,
			nullptr,
			(uint64_t)0
		};

		tuple<vec<double>*, vec<double>*, uint64_t> test_set_cross =
		{
			nullptr,
			nullptr,
			(uint64_t)0
		};

		bool cross_train_mode = use_cross_train_mode(train_set, test_set, _split_set,train_set_cross, test_set_cross);
		bool max_epo_reached = true;

		if (train_set_cross != tuple<vec<double>*, vec<double>*, uint64_t>{ nullptr, nullptr, (uint64_t)0 })
			train_set = train_set_cross;

		if (test_set_cross != tuple<vec<double>*, vec<double>*, uint64_t>{ nullptr, nullptr, (uint64_t)0 })
			test_set = test_set_cross;

		add_info(errors, error_pipe, info(max_epochs, msg_type::count_epo));
		add_info(errors, error_pipe, info(get<(size_t)2>(train_set), msg_type::count_set_1));
		add_info(errors, error_pipe, info(get<(size_t)2>(test_set), msg_type::count_set_2));
		add_info(errors, error_pipe, info(msg_type::batch_mode));

		if (cross_train_mode)
			add_info(errors, error_pipe, info(msg_type::cross_mode));

		uint64_t err_overs = (uint64_t)0;
		double min_test_err = (double)0;
		double prev_test_err = (double)0;

		convergence *= (double)_count;
		uint64_t last_layer = _count - (uint64_t)1;

		uint64_t copy_count = (uint64_t)(omp_get_max_threads() < get<(size_t)2>(train_set)
			? omp_get_max_threads() : get<(size_t)2>(train_set));
		uint64_t cycles = get<(size_t)2>(train_set) / copy_count
			+ (get<(size_t)2>(train_set) % copy_count == (uint64_t)0? (uint64_t)0 : (uint64_t)1);

		light_appx alpha_appxer(alpha_start, alpha_end, max_epochs);
		light_appx speed_appxer(speed_start, speed_end, max_epochs);
		light_appx conv_appxer(max(convergence, (double)1), (double)1, max_epochs);

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

			light_appx speed_lay_appxer(conv_appxer.forward(), speed_epo, _count, (double)0, (double)2);
			selector->reset();

			for (uint64_t j = (uint64_t)0; j < cycles; ++j)
			{
				uint64_t copy_left = j == cycles - (uint64_t)1? get<(size_t)2>(train_set) - j * copy_count : copy_count;

				for (uint64_t num = (uint64_t)0; num < copy_left; ++num)
					selectors[num] = selector->next();

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
						tuple<vec<double>, vec<double>> train_fwd_result =
							move(_layers[k].train_fwd(k == (uint64_t)0
								? get<(size_t)0>(train_set)[num]:activs[k - (uint64_t)1]));

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
							move(_layers[k].train_bwd(grads[k], grads_bias[k]));

						__assume(k_prev >= (uint64_t)0);
						grads[k_prev] = move(get<(size_t)0>(train_bwd_result));
						grads_bias[k_prev] = move(get<(size_t)1>(train_bwd_result));

						for (uint64_t l = (uint64_t)0; l < grads[k_prev].get_size(); ++l)
							grads[k_prev](l) *= derivs[k_prev](l);
					}
				}

				speed_lay_appxer.reset();

				for (uint64_t k = (uint64_t)0; k < _count; ++k)
				{
					double speed = speed_lay_appxer.forward() / (double)copy_left;

					deltas[k] *= alpha_epo;
					deltas_bias[k] *= alpha_epo;

					for (uint64_t num = (uint64_t)0; num < copy_left; ++num)
					{
						uint64_t l = (uint64_t)0;

						__assume(deltas[k].get_size_1() == deltas_bias[k].get_size());
						for (uint64_t m = (uint64_t)0; m < deltas[k].get_size_1(); ++m)
						{
							for (uint64_t n = (uint64_t)0; n < deltas[k].get_size_2(); ++n, ++l)
								deltas[k](l) += arr_grads[num][k](m)* (k == (uint64_t)0
									? get<(size_t)0>(train_set)[selectors[num]]
									: arr_activs[num][k - (uint64_t)1])(n) * speed;

							deltas_bias[k](m) += arr_grads_bias[num][k](m) * speed;
						}
					}

					_layers[k].train_upd(deltas[k], deltas_bias[k]);
				}
			}

			if (cross_train_mode && i % test_freq == (uint64_t)0)
			{
#pragma omp parallel for reduction(+:test_err) num_threads(copy_count)
				for (int64_t j = (int64_t)0; j < get<(size_t)2>(test_set); ++j)
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
			else if (cross_train_mode)
				test_err = prev_test_err;
			else
				test_err = (double)0;

			prev_test_err = test_err;

			train_err /= (double)get<(size_t)2>(train_set);

			add_info(errors, error_pipe, info(train_err, test_err));

			if (i == (uint64_t)0 || test_err < min_test_err)
				min_test_err = test_err;

			if (test_err > min_test_err * (max_error + (double)1))
				++err_overs;

			if (test_err > min_test_err * (max_error * (double)2 + (double)1) || err_overs > max_overs)
			{
				add_info(errors, error_pipe, info(i + (uint64_t)1, msg_type::max_err_reached));
				max_epo_reached = false;

				break;
			}

			if (train_err < min_error)
			{
				add_info(errors, error_pipe, info(i + (uint64_t)1, msg_type::min_err_reached));
				max_epo_reached = false;

				break;
			}
		}

		if (max_epo_reached)
			add_info(errors, error_pipe, info(msg_type::max_epo_reached));

		if (train_set_cross != tuple<vec<double>*, vec<double>*, uint64_t>{ nullptr, nullptr, (uint64_t)0 })
		{
			delete[] get<(size_t)0>(train_set_cross);
			delete[] get<(size_t)1>(train_set_cross);
		}

		if (test_set_cross != tuple<vec<double>*, vec<double>*, uint64_t>{ nullptr, nullptr, (uint64_t)0 })
		{
			delete[] get<(size_t)0>(test_set_cross);
			delete[] get<(size_t)1>(test_set_cross);
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
	}
	else
		throw exception("Wrong data set");
}

void fnn::train_stoch_mode
(
	tuple<const vec<double>*, const vec<double>*, uint64_t> train_set,
	tuple<const vec<double>*, const vec<double>*, uint64_t> test_set,
	list<info>* errors,
	pipe<info>* error_pipe,
	uint64_t max_epochs,
	uint64_t max_overs,
	uint64_t test_freq,
	bool _split_set,
	double convergence,
	double alpha_start,
	double alpha_end,
	double speed_start,
	double speed_end,
	double max_error,
	double min_error
)
{
	if (is_empty())
		throw exception("FNN is not initialized");
	else if (check_set(train_set))
	{
		tuple<vec<double>*, vec<double>*, uint64_t> train_set_cross =
		{
			nullptr,
			nullptr,
			(uint64_t)0
		};

		tuple<vec<double>*, vec<double>*, uint64_t> test_set_cross =
		{
			nullptr,
			nullptr,
			(uint64_t)0
		};

		bool cross_train_mode = use_cross_train_mode(train_set, test_set, _split_set,train_set_cross, test_set_cross);
		bool max_epo_reached = true;

		if (train_set_cross != tuple<vec<double>*, vec<double>*, uint64_t>{ nullptr, nullptr, (uint64_t)0 })
			train_set = train_set_cross;

		if (test_set_cross != tuple<vec<double>*, vec<double>*, uint64_t>{ nullptr, nullptr, (uint64_t)0 })
			test_set = test_set_cross;

		add_info(errors, error_pipe, info(max_epochs, msg_type::count_epo));
		add_info(errors, error_pipe, info(get<(size_t)2>(train_set), msg_type::count_set_1));
		add_info(errors, error_pipe, info(get<(size_t)2>(test_set), msg_type::count_set_2));
		add_info(errors, error_pipe, info(msg_type::stoch_mode));

		if (cross_train_mode)
			add_info(errors, error_pipe, info(msg_type::stoch_mode));

		uint64_t err_overs = (uint64_t)0;
		double min_test_err = (double)0;
		double prev_test_err = (double)0;

		convergence *= (double)_count;
		uint64_t last_layer = _count - (uint64_t)1;

		light_appx alpha_appxer(alpha_start, alpha_end, max_epochs);
		light_appx speed_appxer(speed_start, speed_end, max_epochs);
		light_appx conv_appxer(max(convergence, (double)1), (double)1, max_epochs);

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

			light_appx speed_lay_appxer(conv_appxer.forward(), speed_epo, _count, (double)0, (double)2);
			selector->reset();

			for (uint64_t j = (uint64_t)0; j < get<(size_t)2>(train_set); ++j)
			{
				uint64_t num = selector->next();

				for (uint64_t k = (uint64_t)0; k < _count; ++k)
				{
					tuple<vec<double>, vec<double>> train_fwd_result =
						move(_layers[k].train_fwd(k == (uint64_t)0
							? get<(size_t)0>(train_set)[num]: activs[k - (uint64_t)1]));

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
						move(_layers[k].train_bwd(grads[k], grads_bias[k]));

					__assume(k_prev >= (uint64_t)0);
					grads[k_prev] = move(get<(size_t)0>(train_bwd_result));
					grads_bias[k_prev] = move(get<(size_t)1>(train_bwd_result));

					for (uint64_t l = (uint64_t)0; l < grads[k_prev].get_size(); ++l)
						grads[k_prev](l) *= derivs[k_prev](l);
				}

				speed_lay_appxer.reset();

				for (uint64_t k = (uint64_t)0; k < _count; ++k)
				{
					double speed_lay = speed_lay_appxer.forward();
					uint64_t l = (uint64_t)0;

					__assume(deltas[k].get_size_1() == deltas_bias[k].get_size());
					for (uint64_t m = (uint64_t)0; m < deltas[k].get_size_1(); ++m)
					{
						for (uint64_t n = (uint64_t)0; n < deltas[k].get_size_2(); ++n, ++l)
							deltas[k](l) = deltas[k](l) * alpha_epo + grads[k](m) * (k == (uint64_t)0
								? get<(size_t)0>(train_set)[num]
								: activs[k - (uint64_t)1])(n) * speed_lay;

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
			else if (cross_train_mode)
				test_err = prev_test_err;
			else
				test_err = (double)0;

			prev_test_err = test_err;

			train_err /= (double)get<(size_t)2>(train_set);

			add_info(errors, error_pipe, info(train_err, test_err));

			if (i == (uint64_t)0 || test_err < min_test_err)
				min_test_err = test_err;

			if (test_err > min_test_err * (max_error + (double)1))
				++err_overs;

			if (test_err > min_test_err * (max_error * (double)2 + (double)1) || err_overs > max_overs)
			{
				add_info(errors, error_pipe, info(i + (uint64_t)1, msg_type::max_err_reached));
				max_epo_reached = false;

				break;
			}

			if (train_err < min_error)
			{
				add_info(errors, error_pipe, info(i + (uint64_t)1, msg_type::min_err_reached));
				max_epo_reached = false;

				break;
			}
		}

		if (max_epo_reached)
			add_info(errors, error_pipe, info(msg_type::max_epo_reached));

		if (train_set_cross != tuple<vec<double>*, vec<double>*, uint64_t>{ nullptr, nullptr, (uint64_t)0 })
		{
			delete[] get<(size_t)0>(train_set_cross);
			delete[] get<(size_t)1>(train_set_cross);
		}

		if (test_set_cross != tuple<vec<double>*, vec<double>*, uint64_t>{ nullptr, nullptr, (uint64_t)0 })
		{
			delete[] get<(size_t)0>(test_set_cross);
			delete[] get<(size_t)1>(test_set_cross);
		}

		delete selector;

		delete[] activs;
		delete[] derivs;

		delete[] grads;
		delete[] grads_bias;

		delete[] deltas;
		delete[] deltas_bias;
	}
	else
		throw exception("Wrong data set");
}

inline uint64_t fnn::get_count() const noexcept
{
	return _count;
}

uint64_t fnn::get_param_count() const noexcept
{
	if (is_empty())
		return (uint64_t)0;
	else
	{
		uint64_t result = (uint64_t)0;

		for (uint64_t i = (uint64_t)0; i < _count; ++i)
			result += _layers[i].get_param_count();

		return result;
	}
}

inline bool fnn::is_empty() const noexcept
{
	return _layers == nullptr;
}

fnn& fnn::operator=(const fnn& o)
{
	if (_layers != nullptr)
		delete[] _layers;

	if (o.is_empty())
	{
		_layers = nullptr;
		_count = (uint64_t)0;
	}
	else
	{
		_layers = new fnnl[o._count]();
		_count = o._count;

		for (uint64_t i = (uint64_t)0; i < _count; ++i)
			_layers[i] = o._layers[i];
	}

	return *this;
}

fnn& fnn::operator=(fnn&& o) noexcept
{
	if (_layers != nullptr)
		delete[] _layers;

	_layers = o._layers;
	_count = o._count;

	o._layers = nullptr;
	o._count = (uint64_t)0;

	return *this;
}