#pragma once

#include <cmath>
#include "nn_params.h"

namespace nn_derivs
{
	template <nn_params::nn_activ_t activ>
	inline FLT derivation(FLT x);

	template <>
	inline FLT derivation<nn_params::nn_activ_t::signed_pos>(FLT x)
	{
		return ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::signed_neg>(FLT x)
	{
		return ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::thresh_pos>(FLT x)
	{
		return ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::thresh_neg>(FLT x)
	{
		return ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::step_sym>(FLT x)
	{
		return x >= ((FLT)1) || x <= ((FLT)-1) ? ((FLT)0) : ((FLT)1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::step_pos>(FLT x)
	{
		return x >= ((FLT)1) || x <= ((FLT)0) ? ((FLT)0) : ((FLT)1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::step_neg>(FLT x)
	{
		return x >= ((FLT)0) || x <= ((FLT)-1) ? ((FLT)0) : ((FLT)1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::rad_bas_pos>(FLT x)
	{
		return exp(x * x * ((FLT)-1)) * x * ((FLT)-2);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::rad_bas_neg>(FLT x)
	{
		return exp(x * x * ((FLT)-1)) * x * ((FLT)2);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::sigmoid_log>(FLT x)
	{
		FLT exp_x = exp(x);
		FLT exp_x_1 = exp_x + ((FLT)1);
		return exp_x / (exp_x_1 * exp_x_1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::sigmoid_rat>(FLT x)
	{
		FLT abs_x_1 = abs(x) + ((FLT)1);
		return ((FLT)1) / (abs_x_1 * abs_x_1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::atan>(FLT x)
	{
		return ((FLT)1) / (x * x + ((FLT)1));
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::tanh>(FLT x)
	{
		FLT tanh_x = tanh(x);
		return ((FLT)1) - tanh_x * tanh_x;
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::elu>(FLT x)
	{
		return x >= ((FLT)0) ? ((FLT)1) : exp(x);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::gelu>(FLT x)
	{
		FLT tanh_x = tanh(x);
		return (((FLT)1) - tanh_x * tanh_x) * x + tanh_x + ((FLT)1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::lelu>(FLT x)
	{
		return x >= ((FLT)0) ? ((FLT)1) : ((FLT)0.0625);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::relu>(FLT x)
	{
		return x >= ((FLT)0) ? ((FLT)1) : ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::mish>(FLT x)
	{
		FLT exp_x = exp(x);
		FLT tl = tanh(log(exp_x + ((FLT)1)));
		return (((FLT)1) - tl * tl) * exp_x * x / (exp_x + ((FLT)1)) + tl;
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::swish>(FLT x)
	{
		FLT exp_x = exp(x);
		FLT exp_x_1 = exp_x + ((FLT)1);
		return (exp_x_1 + x) * exp_x / (exp_x_1 * exp_x_1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::softplus>(FLT x)
	{
		FLT exp_x = exp(x);
		return exp_x / (exp_x + ((FLT)1));
	}
}