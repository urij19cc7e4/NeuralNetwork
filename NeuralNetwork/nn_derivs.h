#pragma once

#include <cmath>
#include "nn_params.h"

namespace nn_derivs
{
	template <nn_params::nn_activ_t activ>
	inline FLT derivation(FLT x, FLT sz);

	template <>
	inline FLT derivation<nn_params::nn_activ_t::signed_pos>(FLT x, FLT sz)
	{
		return ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::signed_neg>(FLT x, FLT sz)
	{
		return ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::thresh_pos>(FLT x, FLT sz)
	{
		return ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::thresh_neg>(FLT x, FLT sz)
	{
		return ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::step_sym>(FLT x, FLT sz)
	{
		return x >= ((FLT)1) || x <= ((FLT)-1) ? ((FLT)0) : ((FLT)1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::step_pos>(FLT x, FLT sz)
	{
		return x >= ((FLT)1) || x <= ((FLT)0) ? ((FLT)0) : ((FLT)1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::step_neg>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) || x <= ((FLT)-1) ? ((FLT)0) : ((FLT)1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::rad_bas_pos>(FLT x, FLT sz)
	{
		return exp(x * x * ((FLT)-1)) * x * ((FLT)-2);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::rad_bas_neg>(FLT x, FLT sz)
	{
		return exp(x * x * ((FLT)-1)) * x * ((FLT)2);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::sigmoid_log>(FLT x, FLT sz)
	{
		FLT exp_x = exp(x);
		FLT exp_x_1 = exp_x + ((FLT)1);
		return exp_x / (exp_x_1 * exp_x_1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::sigmoid_rat>(FLT x, FLT sz)
	{
		FLT abs_x_1 = abs(x) + ((FLT)1);
		return ((FLT)1) / (abs_x_1 * abs_x_1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::atan>(FLT x, FLT sz)
	{
		return ((FLT)1) / (x * x + ((FLT)1));
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::tanh>(FLT x, FLT sz)
	{
		FLT _tanh = tanh(x);
		return ((FLT)1) - _tanh * _tanh;
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::elu>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) ? ((FLT)1) : exp(x);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::gelu>(FLT x, FLT sz)
	{
		FLT _tanh = tanh(x);
		return (((FLT)1) - _tanh * _tanh) * x + _tanh + ((FLT)1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::lelu>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) ? ((FLT)1) : ((FLT)0.0625);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::relu>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) ? ((FLT)1) : ((FLT)0);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::mish>(FLT x, FLT sz)
	{
		FLT exp_x = exp(x);
		FLT _tanh = tanh(log(exp_x + ((FLT)1)));
		return (((FLT)1) - _tanh * _tanh) * exp_x * x / (exp_x + ((FLT)1)) + _tanh;
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::swish>(FLT x, FLT sz)
	{
		FLT exp_x = exp(x);
		FLT exp_x_1 = exp_x + ((FLT)1);
		return (exp_x_1 + x) * exp_x / (exp_x_1 * exp_x_1);
	}

	template <>
	inline FLT derivation<nn_params::nn_activ_t::softplus>(FLT x, FLT sz)
	{
		FLT exp_x = exp(x);
		return exp_x / (exp_x + ((FLT)1));
	}
}