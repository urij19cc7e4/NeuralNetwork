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

	enum class nn_convo_t : uint64_t
	{
		many_to_many,
		many_to_one,
		one_to_one,
		__count__
	};

	enum class nn_init_t : uint64_t
	{
		normal,
		uniform,
		__count__
	};

	struct nn_info {};

	struct cnn_info : nn_info
	{
		uint64_t height;
		uint64_t width;
		uint64_t size;
		nn_activ_t activ;
		nn_init_t init;
		FLT scale_x;
		FLT scale_y;
		FLT scale_z;
		nn_convo_t convo;
		bool pool;
	};

	struct fnn_info : nn_info
	{
		uint64_t isize;
		uint64_t osize;
		nn_activ_t activ;
		nn_init_t init;
		FLT scale_x;
		FLT scale_y;
		FLT scale_z;
	};

	struct cnn_2_fnn_info : nn_info
	{
		bool max_pool;
	};

	extern FLT(*activs[(uint64_t)nn_activ_t::__count__])(FLT, FLT);
	extern FLT(*derivs[(uint64_t)nn_activ_t::__count__])(FLT, FLT);

	extern inline FLT activation(FLT x, FLT sx, FLT sy, FLT sz, nn_activ_t activ);
	extern inline FLT derivation(FLT x, FLT sx, FLT sy, FLT sz, nn_activ_t activ);
}