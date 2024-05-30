#pragma once

#include <cmath>
#include "fnn_params.h"

namespace fnn_activs
{
	template <fnn_params::fnn_activ activ>
	inline double activation(double x, double p);

	template <>
	inline double activation<fnn_params::fnn_activ::signed_pos>(double x, double p)
	{
		return (x >= (double)0 ? (double)1 : (double)-1) * p;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::signed_neg>(double x, double p)
	{
		return (x > (double)0 ? (double)1 : (double)-1) * p;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::thresh_pos>(double x, double p)
	{
		return (x >= (double)0 ? (double)1 : (double)0) * p;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::thresh_neg>(double x, double p)
	{
		return (x > (double)0 ? (double)1 : (double)0) * p;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::step_sym>(double x, double p)
	{
		return (x >= (double)1 ? (double)1 : x <= (double)-1 ? (double)-1 : x) * p;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::step_pos>(double x, double p)
	{
		return (x >= (double)1 ? (double)1 : x <= (double)0 ? (double)0 : x) * p;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::step_neg>(double x, double p)
	{
		return (x >= (double)0 ? (double)0 : x <= (double)-1 ? (double)-1 : x) * p;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::rad_bas_pos>(double x, double p)
	{
		return (double)1 * exp((double)-1 * x * x / abs(p));
	}

	template <>
	inline double activation<fnn_params::fnn_activ::rad_bas_neg>(double x, double p)
	{
		return (double)-1 * exp((double)-1 * x * x / abs(p));
	}

	template <>
	inline double activation<fnn_params::fnn_activ::sigmoid_log>(double x, double p)
	{
		return (double)1 / ((double)1 + exp((double)-1 * x * p));
	}

	template <>
	inline double activation<fnn_params::fnn_activ::sigmoid_rat>(double x, double p)
	{
		return x / (abs(x) + p);
	}

	template <>
	inline double activation<fnn_params::fnn_activ::atan>(double x, double p)
	{
		return atan(x * p);
	}

	template <>
	inline double activation<fnn_params::fnn_activ::tanh>(double x, double p)
	{
		return tanh(x * p);
	}

	template <>
	inline double activation<fnn_params::fnn_activ::elu>(double x, double p)
	{
		return (x >= (double)0 ? x : exp(x) - 1) * p;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::gelu>(double x, double p)
	{
		return ((double)1 + tanh((double)0.797884560802865355
			* ((double)1 + (double)0.044715 * x * x * p) * x)) * x;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::lelu>(double x, double p)
	{
		return x >= (double)0 ? x : x * p;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::relu>(double x, double p)
	{
		return x >= (double)0 ? x * p : (double)0;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::mish>(double x, double p)
	{
		return tanh(log((double)1 + exp(x * p)) / p) * x;
	}

	template <>
	inline double activation<fnn_params::fnn_activ::swish>(double x, double p)
	{
		return x / ((double)1 + exp((double)-1 * x * p));
	}

	template <>
	inline double activation<fnn_params::fnn_activ::softplus>(double x, double p)
	{
		return log((double)1 + exp(x * p)) / p;
	}
}