#pragma once

#include <cmath>
#include "nn_params.h"

namespace nn_derivs
{
	template <nn_params::nn_activ_t activ>
	inline double derivation(double x, double y, double p);

	template <>
	inline double derivation<nn_params::nn_activ_t::signed_pos>(double x, double y, double p)
	{
		return (double)0;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::signed_neg>(double x, double y, double p)
	{
		return (double)0;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::thresh_pos>(double x, double y, double p)
	{
		return (double)0;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::thresh_neg>(double x, double y, double p)
	{
		return (double)0;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::step_sym>(double x, double y, double p)
	{
		return x >= (double)1 || x <= (double)-1 ? (double)0 : p;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::step_pos>(double x, double y, double p)
	{
		return x >= (double)1 || x <= (double)0 ? (double)0 : p;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::step_neg>(double x, double y, double p)
	{
		return x >= (double)0 || x <= (double)-1 ? (double)0 : p;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::rad_bas_pos>(double x, double y, double p)
	{
		return (double)2 * x * y / abs(p);
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::rad_bas_neg>(double x, double y, double p)
	{
		return (double)2 * x * y / abs(p);
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::sigmoid_log>(double x, double y, double p)
	{
		return ((double)1 - y) * y * p;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::sigmoid_rat>(double x, double y, double p)
	{
		double abs_x_plus_p = abs(x) + p;
		return p / (abs_x_plus_p * abs_x_plus_p);
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::atan>(double x, double y, double p)
	{
		double x_mult_p = x * p;
		return p / ((double)1 + x_mult_p * x_mult_p);
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::tanh>(double x, double y, double p)
	{
		return ((double)1 - y * y) * p;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::elu>(double x, double y, double p)
	{
		return (x >= (double)0 ? (double)1 : exp(x)) * p;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::gelu>(double x, double y, double p)
	{
		if (x == (double)0)
			return (double)1;
		else
		{
			double z = y / x;
			return ((double)1 + (double)0.797884560802865355 * ((double)1 /* √(2 / π) */
				+ (double)3 * (double)0.044715 * x * x * p) * ((double)2 - z) * x) * z;
		}
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::lelu>(double x, double y, double p)
	{
		return x >= (double)0 ? (double)1 : p;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::relu>(double x, double y, double p)
	{
		return x >= (double)0 ? p : (double)0;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::mish>(double x, double y, double p)
	{
		if (x == (double)0)
			return tanh((double)0.693147180559945309 / p); /* ln(2) */
		else
		{
			double exp_x_mult_p = exp(x * p), z = y / x;
			return (exp_x_mult_p * x / ((double)1 + exp_x_mult_p)) * ((double)1 - z * z) + z;
		}
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::swish>(double x, double y, double p)
	{
		if (x == (double)0)
			return (double)0.5;
		else
			return ((double)1 + (x - y) * p) * y / x;
	}

	template <>
	inline double derivation<nn_params::nn_activ_t::softplus>(double x, double y, double p)
	{
		double exp_x_mult_p = exp(x * p);
		return exp_x_mult_p / ((double)1 + exp_x_mult_p);
	}
}