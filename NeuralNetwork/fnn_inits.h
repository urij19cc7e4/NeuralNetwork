#pragma once

#include <cmath>
#include <cstdint>
#include <random>

#include "fnn_params.h"

namespace fnn_inits
{
	template <fnn_params::fnn_activ activ, fnn_params::fnn_init_t init_t>
	inline fnn_params::fnn_init<init_t> initialization(uint64_t isize, uint64_t osize, double param);

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::signed_pos, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::signed_pos, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::signed_neg, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::signed_neg, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::thresh_pos, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::thresh_pos, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::thresh_neg, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::thresh_neg, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::step_sym, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::step_sym, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::step_pos, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::step_pos, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::step_neg, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::step_neg, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::rad_bas_pos, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::rad_bas_pos, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::rad_bas_neg, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::rad_bas_neg, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::sigmoid_log, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::sigmoid_log, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::sigmoid_rat, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::sigmoid_rat, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::atan, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::atan, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::tanh, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::tanh, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::elu, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::elu, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::gelu, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::gelu, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::lelu, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0,
			sqrt((double)2 / ((double)isize * ((double)1 + param * param))) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::lelu, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / ((double)isize * ((double)1 + param * param))),
			(double)-1 * sqrt((double)6 / ((double)isize * ((double)1 + param * param))) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::relu, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::relu, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::mish, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::mish, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::swish, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::swish, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::normal>
		initialization<fnn_params::fnn_activ::softplus, fnn_params::fnn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::normal>{ (double)0, sqrt((double)2 / (double)(isize + osize)) };
	}

	template <>
	inline fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>
		initialization<fnn_params::fnn_activ::softplus, fnn_params::fnn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return fnn_params::fnn_init<fnn_params::fnn_init_t::uniform>{ sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize)) };
	}

	template <fnn_params::fnn_activ activ, fnn_params::fnn_init_t init_t>
	class initializer;

	template <fnn_params::fnn_activ activ>
	class initializer<activ, fnn_params::fnn_init_t::normal>
	{
	private:
		std::normal_distribution<double> distributor;

	public:
		initializer() = delete;

		initializer(uint64_t isize, uint64_t osize, double param = (double)1)
		{
			fnn_params::fnn_init<fnn_params::fnn_init_t::normal> params =
				initialization<activ, fnn_params::fnn_init_t::normal>(isize, osize, param);
			distributor = std::normal_distribution<double>(params.mean, params.sigm);
		}

		initializer(const initializer& o) = delete;

		initializer(initializer&& o) = delete;

		~initializer() {}

		template <typename URBG>
		double operator()(URBG& engine)
		{
			return distributor(engine);
		}
	};

	template <fnn_params::fnn_activ activ>
	class initializer<activ, fnn_params::fnn_init_t::uniform>
	{
	private:
		std::uniform_real_distribution<double> distributor;

	public:
		initializer() = delete;

		initializer(uint64_t isize, uint64_t osize, double param = (double)1)
		{
			fnn_params::fnn_init<fnn_params::fnn_init_t::uniform> params =
				initialization<activ, fnn_params::fnn_init_t::uniform>(isize, osize, param);
			distributor = std::uniform_real_distribution<double>(params.min, params.max);
		}

		initializer(const initializer& o) = delete;

		initializer(initializer&& o) = delete;

		~initializer() {}

		template <typename URBG>
		double operator()(URBG& engine)
		{
			return distributor(engine);
		}
	};
}