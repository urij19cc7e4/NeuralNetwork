#pragma once

#include <cstdint>

namespace nn_params
{
	enum class nn_activ_t : uint8_t
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

	enum class nn_init_t : uint8_t
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
	};
}