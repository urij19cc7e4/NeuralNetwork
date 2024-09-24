#pragma once

#include <cmath>
#include "nn_params.h"

namespace nn_activs
{
	template <nn_params::nn_activ_t activ>
	inline FLT activation(FLT x);

	template <>
	inline FLT activation<nn_params::nn_activ_t::signed_pos>(FLT x)
	{
		return x >= ((FLT)0) ? ((FLT)1) : ((FLT)-1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::signed_neg>(FLT x)
	{
		return x > ((FLT)0) ? ((FLT)1) : ((FLT)-1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::thresh_pos>(FLT x)
	{
		return x >= ((FLT)0) ? ((FLT)1) : ((FLT)0);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::thresh_neg>(FLT x)
	{
		return x > ((FLT)0) ? ((FLT)1) : ((FLT)0);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::step_sym>(FLT x)
	{
		return x >= ((FLT)1) ? ((FLT)1) : x <= ((FLT)-1) ? ((FLT)-1) : x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::step_pos>(FLT x)
	{
		return x >= ((FLT)1) ? ((FLT)1) : x <= ((FLT)0) ? ((FLT)0) : x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::step_neg>(FLT x)
	{
		return x >= ((FLT)0) ? ((FLT)0) : x <= ((FLT)-1) ? ((FLT)-1) : x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::rad_bas_pos>(FLT x)
	{
		return exp(x * x * ((FLT)-1)) * ((FLT)1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::rad_bas_neg>(FLT x)
	{
		return exp(x * x * ((FLT)-1)) * ((FLT)-1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::sigmoid_log>(FLT x)
	{
		return ((FLT)1) / (exp(x * ((FLT)-1)) + ((FLT)1));
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::sigmoid_rat>(FLT x)
	{
		return x / (abs(x) + ((FLT)1));
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::atan>(FLT x)
	{
		return atan(x);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::tanh>(FLT x)
	{
		return tanh(x);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::elu>(FLT x)
	{
		return x >= ((FLT)0) ? x : exp(x) - ((FLT)1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::gelu>(FLT x)
	{
		return (tanh(x) + ((FLT)1)) * x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::lelu>(FLT x)
	{
		return x >= ((FLT)0) ? x : x * ((FLT)0.0625);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::relu>(FLT x)
	{
		return x >= ((FLT)0) ? x : ((FLT)0);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::mish>(FLT x)
	{
		return tanh(log(exp(x) + ((FLT)1))) * x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::swish>(FLT x)
	{
		return x / (exp(x * ((FLT)-1)) + ((FLT)1));
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::softplus>(FLT x)
	{
		return log(exp(x) + ((FLT)1));
	}
}