#pragma once

#include <cstdint>

typedef double FLT;

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

	extern FLT (*activs[(uint64_t)nn_activ_t::__count__])(FLT, FLT);
	extern FLT (*derivs[(uint64_t)nn_activ_t::__count__])(FLT, FLT);

	extern inline FLT activation(FLT x, FLT sx, FLT sy, FLT sz, nn_activ_t activ);
	extern inline FLT derivation(FLT x, FLT sx, FLT sy, FLT sz, nn_activ_t activ);
}