#pragma once

#include <cmath>
#include "nn_params.h"

namespace nn_activs
{
	template <nn_params::nn_activ_t activ>
	inline FLT activation(FLT x, FLT sz);

	template <>
	inline FLT activation<nn_params::nn_activ_t::signed_pos>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) ? ((FLT)1) : ((FLT)-1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::signed_neg>(FLT x, FLT sz)
	{
		return x > ((FLT)0) ? ((FLT)1) : ((FLT)-1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::thresh_pos>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) ? ((FLT)1) : ((FLT)0);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::thresh_neg>(FLT x, FLT sz)
	{
		return x > ((FLT)0) ? ((FLT)1) : ((FLT)0);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::step_sym>(FLT x, FLT sz)
	{
		return x >= ((FLT)1) ? ((FLT)1) : x <= ((FLT)-1) ? ((FLT)-1) : x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::step_pos>(FLT x, FLT sz)
	{
		return x >= ((FLT)1) ? ((FLT)1) : x <= ((FLT)0) ? ((FLT)0) : x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::step_neg>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) ? ((FLT)0) : x <= ((FLT)-1) ? ((FLT)-1) : x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::rad_bas_pos>(FLT x, FLT sz)
	{
		return exp(x * x * ((FLT)-1)) * ((FLT)1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::rad_bas_neg>(FLT x, FLT sz)
	{
		return exp(x * x * ((FLT)-1)) * ((FLT)-1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::sigmoid_log>(FLT x, FLT sz)
	{
		return ((FLT)1) / (exp(x * ((FLT)-1)) + ((FLT)1));
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::sigmoid_rat>(FLT x, FLT sz)
	{
		return x / (abs(x) + ((FLT)1));
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::atan>(FLT x, FLT sz)
	{
		return atan(x);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::tanh>(FLT x, FLT sz)
	{
		return tanh(x);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::elu>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) ? x : exp(x) - ((FLT)1);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::gelu>(FLT x, FLT sz)
	{
		return (tanh(x) + ((FLT)1)) * x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::lelu>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) ? x : x * ((FLT)0.0625);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::relu>(FLT x, FLT sz)
	{
		return x >= ((FLT)0) ? x : ((FLT)0);
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::mish>(FLT x, FLT sz)
	{
		return tanh(log(exp(x) + ((FLT)1))) * x;
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::swish>(FLT x, FLT sz)
	{
		return x / (exp(x * ((FLT)-1)) + ((FLT)1));
	}

	template <>
	inline FLT activation<nn_params::nn_activ_t::softplus>(FLT x, FLT sz)
	{
		return log(exp(x) + ((FLT)1));
	}
}