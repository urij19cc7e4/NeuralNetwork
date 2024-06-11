#pragma once

#include <cstdint>
#include <random>

namespace nn_params
{
	enum class nn_activ_t : uint64_t
	{
		signed_pos,
		signed_neg,
		thresh_pos,
		thresh_neg,
		step_sym,
		step_pos,
		step_neg,
		rad_bas_pos,
		rad_bas_neg,
		sigmoid_log,
		sigmoid_rat,
		atan,
		tanh,
		elu,
		gelu,
		lelu,
		relu,
		mish,
		swish,
		softplus,
		__count__
	};

	enum class nn_init_t : uint64_t
	{
		normal,
		uniform,
		__count__
	};

	template <nn_init_t init>
	struct nn_init;

	template <>
	struct nn_init<nn_init_t::normal>
	{
		double mean;
		double sigm;
	};

	template <>
	struct nn_init<nn_init_t::uniform>
	{
		double max;
		double min;
	};

	struct fnn_layer_info
	{
		uint64_t isize;
		uint64_t osize;
		nn_activ_t activ;
		nn_init_t init;
		double param;
	};

	extern double (*activs[(uint64_t)nn_activ_t::__count__])(double, double);
	extern double (*derivs[(uint64_t)nn_activ_t::__count__])(double, double, double);

	extern nn_init<nn_init_t::normal> (*inits_normal[(uint64_t)nn_activ_t::__count__])(uint64_t, uint64_t, double);
	extern nn_init<nn_init_t::uniform> (*inits_uniform[(uint64_t)nn_activ_t::__count__])(uint64_t, uint64_t, double);

	extern inline double activation(double x, double p, nn_activ_t activ);
	extern inline double derivation(double x, double y, double p, nn_activ_t activ);

	extern inline nn_init<nn_init_t::normal> init_normal(uint64_t isize, uint64_t osize, double param, nn_activ_t activ);
	extern inline nn_init<nn_init_t::uniform> init_uniform(uint64_t isize, uint64_t osize, double param, nn_activ_t activ);

	class rand_init_base
	{
	protected:
		std::mt19937_64 _rand_gen;

	public:
		rand_init_base() : _rand_gen((std::random_device())()) {}

		rand_init_base(const rand_init_base& o) = delete;

		rand_init_base(rand_init_base&& o) = delete;

		virtual ~rand_init_base() {}

		virtual double operator()() = 0;
	};

	template <nn_init_t init>
	class rand_init;

	template <>
	class rand_init<nn_init_t::normal> : public rand_init_base
	{
	protected:
		std::normal_distribution<double> _distributor;

	public:
		rand_init() = delete;

		rand_init(uint64_t isize, uint64_t osize, double param, nn_activ_t activ)
			: rand_init_base(), _distributor()
		{
			nn_init<nn_init_t::normal> params = init_normal(isize, osize, param, activ);

			_distributor = std::normal_distribution<double>(params.mean, params.sigm);
		}

		rand_init(const rand_init& o) = delete;

		rand_init(rand_init&& o) = delete;

		virtual ~rand_init() {}

		virtual double operator()() override
		{
			return _distributor(this->_rand_gen);
		}
	};

	template <>
	class rand_init<nn_init_t::uniform> : public rand_init_base
	{
	protected:
		std::uniform_real_distribution<double> _distributor;

	public:
		rand_init() = delete;

		rand_init(uint64_t isize, uint64_t osize, double param, nn_activ_t activ)
			: rand_init_base(), _distributor()
		{
			nn_init<nn_init_t::uniform> params = init_uniform(isize, osize, param, activ);

			_distributor = std::uniform_real_distribution<double>(params.min, params.max);
		}

		rand_init(const rand_init& o) = delete;

		rand_init(rand_init&& o) = delete;

		virtual ~rand_init() {}

		virtual double operator()() override
		{
			return _distributor(this->_rand_gen);
		}
	};
}