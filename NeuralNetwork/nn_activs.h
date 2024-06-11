#pragma once

#include <cmath>
#include "nn_params.h"

namespace nn_activs
{
	template <nn_params::nn_activ_t activ>
	inline double activation(double x, double p);

	template <>
	inline double activation<nn_params::nn_activ_t::signed_pos>(double x, double p)
	{
		return (x >= (double)0 ? (double)1 : (double)-1) * p;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::signed_neg>(double x, double p)
	{
		return (x > (double)0 ? (double)1 : (double)-1) * p;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::thresh_pos>(double x, double p)
	{
		return (x >= (double)0 ? (double)1 : (double)0) * p;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::thresh_neg>(double x, double p)
	{
		return (x > (double)0 ? (double)1 : (double)0) * p;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::step_sym>(double x, double p)
	{
		return (x >= (double)1 ? (double)1 : x <= (double)-1 ? (double)-1 : x) * p;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::step_pos>(double x, double p)
	{
		return (x >= (double)1 ? (double)1 : x <= (double)0 ? (double)0 : x) * p;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::step_neg>(double x, double p)
	{
		return (x >= (double)0 ? (double)0 : x <= (double)-1 ? (double)-1 : x) * p;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::rad_bas_pos>(double x, double p)
	{
		return (double)1 * exp((double)-1 * x * x / abs(p));
	}

	template <>
	inline double activation<nn_params::nn_activ_t::rad_bas_neg>(double x, double p)
	{
		return (double)-1 * exp((double)-1 * x * x / abs(p));
	}

	template <>
	inline double activation<nn_params::nn_activ_t::sigmoid_log>(double x, double p)
	{
		return (double)1 / ((double)1 + exp((double)-1 * x * p));
	}

	template <>
	inline double activation<nn_params::nn_activ_t::sigmoid_rat>(double x, double p)
	{
		return x / (abs(x) + p);
	}

	template <>
	inline double activation<nn_params::nn_activ_t::atan>(double x, double p)
	{
		return atan(x * p);
	}

	template <>
	inline double activation<nn_params::nn_activ_t::tanh>(double x, double p)
	{
		return tanh(x * p);
	}

	template <>
	inline double activation<nn_params::nn_activ_t::elu>(double x, double p)
	{
		return (x >= (double)0 ? x : exp(x) - 1) * p;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::gelu>(double x, double p)
	{
		return ((double)1 + tanh((double)0.797884560802865355 /* √(2 / π) */
			* ((double)1 + (double)0.044715 * x * x * p) * x)) * x;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::lelu>(double x, double p)
	{
		return x >= (double)0 ? x : x * p;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::relu>(double x, double p)
	{
		return x >= (double)0 ? x * p : (double)0;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::mish>(double x, double p)
	{
		return tanh(log((double)1 + exp(x * p)) / p) * x;
	}

	template <>
	inline double activation<nn_params::nn_activ_t::swish>(double x, double p)
	{
		return x / ((double)1 + exp((double)-1 * x * p));
	}

	template <>
	inline double activation<nn_params::nn_activ_t::softplus>(double x, double p)
	{
		return log((double)1 + exp(x * p)) / p;
	}
}