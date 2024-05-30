#pragma once

#include <cstdint>
#include <exception>
#include <random>

#include "rfm.h"
#include "fnn_params.h"
#include "fnn_activs.h"
#include "fnn_derivs.h"
#include "fnn_inits.h"

using namespace std;

using namespace arithmetic;
using namespace fnn_params;
using namespace fnn_activs;
using namespace fnn_derivs;
using namespace fnn_inits;

template <fnn_activ activ, fnn_init_t init_t>
class fnnl
{
private:
	mtx<double> _weights;
	double _bias_weight;
	double _activ_param;

public:
	fnnl() noexcept
		: _weights(), _bias_weight((double)0), _activ_param((double)1) {}

	fnnl(uint64_t isize, uint64_t osize, double activ_param = (double)1)
		: _weights(osize, isize), _bias_weight((double)0), _activ_param(activ_param)
	{
		if (isize == (uint64_t)0 || osize == (uint64_t)0)
			throw exception("FNN layer initialized with null");
		else
		{
			random_device rand_dev;
			mt19937_64 rand_gen(rand_dev());
			initializer<activ, init_t> dstr(isize, osize, activ_param);

			for (uint64_t i = (uint64_t)0; i < isize * osize; ++i)
				_weights(i) = dstr(rand_gen);
		}
	}

	fnnl(const mtx<double>& weights, double bias_weight = (double)0, double activ_param = (double)1)
		: _weights(weights), _bias_weight(bias_weight), _activ_param(activ_param) {}

	fnnl(mtx<double>&& weights, double bias_weight = (double)0, double activ_param = (double)1) noexcept
		: _weights(move(weights)), _bias_weight(bias_weight), _activ_param(activ_param) {}

	fnnl(const fnnl& o)
		: _weights(o._weights), _bias_weight(o._bias_weight), _activ_param(o._activ_param) {}

	fnnl(fnnl&& o) noexcept
		: _weights(move(o._weights)), _bias_weight(o._bias_weight), _activ_param(o._activ_param) {}

	~fnnl() {}

	vec<double> pass_bwd(const vec<double>& output) const
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

	vec<double> pass_fwd(const vec<double>& input) const
	{
		if (is_empty())
			throw exception("FNN layer is not initialized");
		else if (_weights.get_size_2() == input.get_size())
			return apply_multiply<double>(_weights, input,
				[=](double x) -> double { return activation<activ>(_bias_weight + x, _activ_param); });
		else
			throw exception("Given vector size and FNN layer input vector size do not match");
	}

	vec<double> train_bwd(const vec<double>& deltas) const
	{
		if (is_empty())
			throw exception("FNN layer is not initialized");
		else if (_weights.get_size_1() == deltas.get_size())
			return deltas * _weights;
		else
			throw exception("Given vector size and FNN layer output vector size do not match");
	}

	tuple<vec<double>, vec<double>> train_fwd(const vec<double>& input) const
	{
		if (is_empty())
			throw exception("FNN layer is not initialized");
		else if (_weights.get_size_2() == input.get_size())
			return apply_multiply<double>(_weights, input,
				[=](double x) -> double { return activation<activ>(_bias_weight + x, _activ_param); },
				[=](double x, double y) -> double { return derivation<activ>(_bias_weight + x, y, _activ_param); });
		else
			throw exception("Given vector size and FNN layer input vector size do not match");
	}

	void train_upd(const vec<double>& deltas, const vec<double>& input, double speed)
	{
		if (is_empty())
			throw exception("FNN layer is not initialized");
		else if (_weights.get_size_1() == deltas.get_size() && _weights.get_size_2() == input.get_size())
		{
			uint64_t k = (uint64_t)0;

			for (uint64_t i = (uint64_t)0; i < _weights.get_size_1(); ++i)
				for (uint64_t j = (uint64_t)0; j < _weights.get_size_2(); ++j, ++k)
					_weights(k) -= deltas(i) * input(j) * speed;
		}
		else
			throw exception("Given vector size and FNN layer input vector size do not match");
	}

	uint64_t get_isize() const noexcept
	{
		return _weights.get_size_2();
	}

	uint64_t get_osize() const noexcept
	{
		return _weights.get_size_1();
	}

	double get_param() const noexcept
	{
		return _activ_param;
	}

	bool is_empty() const noexcept
	{
		return _weights.is_empty();
	}

	fnnl& operator=(const fnnl& o)
	{
		_weights = o._weights;
		_bias_weight = o._bias_weight;
		_activ_param = o._activ_param;

		return *this;
	}

	fnnl& operator=(fnnl&& o) noexcept
	{
		_weights = move(o._weights);
		_bias_weight = o._bias_weight;
		_activ_param = o._activ_param;

		return *this;
	}
};

template <fnn_activ activ, fnn_init_t init_t>
class fnn
{
private:
	fnnl<activ, init_t>* _layers;
	uint64_t _count;

public:
	fnn() noexcept : _layers(nullptr), _count((uint64_t)0) {}

	fnn(uint64_t isize, uint64_t osize, uint64_t ssize, uint64_t count, double param = (double)1) : _count(count)
	{
		if (isize == (uint64_t)0 || osize == (uint64_t)0
			|| (ssize == (uint64_t)0 && count > (uint64_t)1) || count == (uint64_t)0)
		{
			_layers = nullptr;
			_count = (uint64_t)0;

			throw exception("FNN layer(s) initialized with null(s)");
		}
		else
		{
			_layers = new fnnl<activ, init_t>[_count]();

			for (uint64_t i = (uint64_t)0; i < _count; ++i)
				_layers[i] = fnnl<activ, init_t>(
					i == (uint64_t)0 ? isize : ssize,
					i == _count - (uint64_t)1 ? osize : ssize, param);
		}
	}

	fnn(const fnn_layer_info* info, uint64_t count, double param = (double)1) : _count(count)
	{
		if (info == nullptr || count == (uint64_t)0)
		{
			_layers = nullptr;
			_count = (uint64_t)0;

			throw exception("FNN layer(s) initialized with null(s)");
		}
		else
		{
			_layers = new fnnl<activ, init_t>[_count]();

			uint64_t prev_size = info[0].isize;
			for (uint64_t i = (uint64_t)0; i < _count; ++i)
			{
				if (info[i].isize == prev_size)
					_layers[i] = fnnl<activ, init_t>(info[i].isize, info[i].osize, param);
				else
				{
					if (_layers != nullptr)
						delete[] _layers;

					_layers = nullptr;
					_count = (uint64_t)0;

					throw exception("FNN layers sizes do not match");
				}

				prev_size = info[i].osize;
			}
		}
	}

	fnn(initializer_list<uint64_t> sizes, double param = (double)1)
		: _count(sizes.size() <= (uint64_t)1 ? (uint64_t)0 : sizes.size() - (uint64_t)1)
	{
		if (_count == (uint64_t)0)
		{
			_layers = nullptr;
			_count = (uint64_t)0;

			throw exception("FNN layer(s) initialized with null(s)");
		}
		else
		{
			_layers = new fnnl<activ, init_t>[_count]();

			const uint64_t* sizes_arr = sizes.begin();
			for (uint64_t i = (uint64_t)0; i < _count;++i)
				_layers[i] = fnnl<activ, init_t>(sizes_arr[i], sizes_arr[i+(uint64_t)1], param);
		}
	}

	fnn(const fnn& o) : _count(o._count)
	{
		if (o.is_empty())
		{
			_layers = nullptr;
			_count = (uint64_t)0;
		}
		else
		{
			_layers = new fnnl<activ, init_t>[_count]();

			for (uint64_t i = (uint64_t)0; i < _count; ++i)
				_layers[i] = o._layers[i];
		}
	}

	fnn(fnn&& o) noexcept : _layers(o._layers), _count(o._count)
	{
		o._layers = nullptr;
		o._count = (uint64_t)0;
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
			vec<double> result = _layers[(uint64_t)0].pass_fwd(input);

			for (uint64_t i = (uint64_t)1; i < _count; ++i)
				result = _layers[i].pass_fwd(result);

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
				input = _layers[i].pass_fwd(input);

			return input;
		}
		else
			throw exception("Given vector size and FNN input vector size do not match");
	}

	vec<double> train(const vec<double>* input, const vec<double>* output, uint64_t batch_size,
		uint64_t max_epochs = (uint64_t)1000, double min_errors = (double)1e-6, double speed = (double)0.05)
	{
		uint64_t last_layer = _count - (uint64_t)1;

		if (is_empty())
			throw exception("FNN is not initialized");
		else if (input == nullptr || output == nullptr || batch_size == (uint64_t)0)
			throw exception("Given empty train batch");
		else
		{
			for (uint64_t i = (uint64_t)0; i < batch_size; ++i)
				if (input[i].is_empty() || _layers[(uint64_t)0].get_isize() != input[i].get_size()
					|| output[i].is_empty() || _layers[last_layer].get_osize() != output[i].get_size())
					throw exception("Given wrong train batch");

			vec<double> errors(max_epochs);

			vec<double>* activs = new vec<double>[_count]();
			vec<double>* derivs = new vec<double>[_count]();
			vec<double>* deltas = new vec<double>[_count]();

			deltas[last_layer] = vec<double>(_layers[last_layer].get_osize());

			for (uint64_t i = (uint64_t)0; i < max_epochs; ++i)
			{
				double error = (double)0;

				for (uint64_t j = (uint64_t)0; j < batch_size; ++j)
				{
					for (uint64_t k = (uint64_t)0; k < _count; ++k)
					{
						uint64_t k_prev = k - (uint64_t)1;
						tuple<vec<double>, vec<double>> train_fwd_result =
							_layers[k].train_fwd(k == (uint64_t)0 ? input[j] : activs[k_prev]);

						activs[k] = move(get<(size_t)0>(train_fwd_result));
						derivs[k] = move(get<(size_t)1>(train_fwd_result));
					}

					for (uint64_t k = (uint64_t)0; k < _layers[last_layer].get_osize(); ++k)
					{
						double difference = activs[last_layer](k) - output[j](k);

						deltas[last_layer](k) = derivs[last_layer](k) * difference;
						error += difference * difference;
					}

					for (uint64_t k = last_layer; k > (uint64_t)0; --k)
					{
						uint64_t k_prev = k - (uint64_t)1;
						deltas[k_prev] = _layers[k].train_bwd(deltas[k]);

						for (uint64_t l = (uint64_t)0; l < deltas[k_prev].get_size(); ++l)
							deltas[k_prev](l) *= derivs[k_prev](l);
					}

					for (uint64_t k = (uint64_t)0; k < _count; ++k)
						_layers[k].train_upd(deltas[k],
							k == (uint64_t)0 ? input[j] : activs[k - (uint64_t)1], speed);
				}

				if (error < min_errors)
				{
					vec<double> new_errors(i + (uint64_t)1);

					for (uint64_t j = (uint64_t)0; j < i; ++j)
						new_errors(j) = errors(j);
					new_errors(i) = error;

					errors = move(new_errors);
					break;
				}
				else
					errors(i) = error;
			}

			delete[] activs;
			delete[] derivs;
			delete[] deltas;

			return errors;
		}
	}

	uint64_t get_count() const noexcept
	{
		return _count;
	}

	bool is_empty() const noexcept
	{
		return _layers == nullptr || _count == (uint64_t)0;
	}

	fnn& operator=(const fnn& o)
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
			_layers = new fnnl<activ, init_t>[o._count]();
			_count = o._count;

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

		o._layers = nullptr;
		o._count = (uint64_t)0;

		return *this;
	}
};