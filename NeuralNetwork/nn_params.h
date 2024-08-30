#pragma once

#include <cstdint>

typedef double FLT;

class nn;

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

	struct nn_info
	{
	public:
		virtual nn* create_new() const = 0;
	};

	struct cnn_info : nn_info
	{
	public:
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

		cnn_info(uint64_t height, uint64_t width, uint64_t size, nn_activ_t activ, nn_init_t init,
			FLT scale_x, FLT scale_y, FLT scale_z, nn_convo_t convo, bool pool);

		virtual nn* create_new() const;
	};

	struct fnn_info : nn_info
	{
	public:
		uint64_t isize;
		uint64_t osize;
		nn_activ_t activ;
		nn_init_t init;
		FLT scale_x;
		FLT scale_y;
		FLT scale_z;

		fnn_info(uint64_t isize, uint64_t osize, nn_activ_t activ, nn_init_t init,
			FLT scale_x, FLT scale_y, FLT scale_z);

		virtual nn* create_new() const;
	};

	struct cnn_2_fnn_info : nn_info
	{
	public:
		bool max_pool;

		cnn_2_fnn_info(bool max_pool);

		virtual nn* create_new() const;
	};

	extern FLT (*activs[(uint64_t)nn_activ_t::__count__])(FLT, FLT);
	extern FLT (*derivs[(uint64_t)nn_activ_t::__count__])(FLT, FLT);

	extern inline FLT activation(FLT x, FLT sx, FLT sy, FLT sz, nn_activ_t activ);
	extern inline FLT derivation(FLT x, FLT sx, FLT sy, FLT sz, nn_activ_t activ);
}