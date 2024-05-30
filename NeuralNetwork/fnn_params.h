#pragma once

#include <cstdint>

namespace fnn_params
{
	enum class fnn_activ : uint8_t
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

	enum class fnn_init_t : uint8_t
	{
		normal,
		uniform,
		__count__
	};

	template <fnn_init_t init>
	struct fnn_init;

	template <>
	struct fnn_init<fnn_init_t::normal>
	{
		double mean;
		double sigm;
	};

	template <>
	struct fnn_init<fnn_init_t::uniform>
	{
		double max;
		double min;
	};

	struct fnn_layer_info
	{
		uint64_t isize;
		uint64_t osize;
	};
}