#pragma once

#include <cstdint>

typedef float FLT;

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

	struct nn_activ_params
	{
	public:
		nn_activ_t activ_type;

		FLT scale_x;
		FLT scale_y;
		FLT shift_x;
		FLT shift_y;

		nn_activ_params() = delete;
		nn_activ_params
		(
			nn_activ_t activ_type,
			FLT scale_x = (FLT)1,
			FLT scale_y = (FLT)1,
			FLT shift_x = (FLT)0,
			FLT shift_y = (FLT)0
		) noexcept;
		nn_activ_params(const nn_activ_params& o) = default;
		nn_activ_params(nn_activ_params&& o) = default;
		~nn_activ_params() = default;
	};

	extern FLT (*activs[(uint64_t)nn_activ_t::__count__])(FLT);
	extern FLT (*derivs[(uint64_t)nn_activ_t::__count__])(FLT);

	extern inline FLT activation(FLT x, const nn_activ_params& params);
	extern inline FLT derivation(FLT x, const nn_activ_params& params);
}