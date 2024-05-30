#pragma once

#include <cmath>
#include "fnn_params.h"

namespace fnn_derivs
{
	template <fnn_params::fnn_activ activ>
	inline double derivation(double x, double y, double p);

	template <>
	inline double derivation<fnn_params::fnn_activ::signed_pos>(double x, double y, double p)
	{
		return (double)0;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::signed_neg>(double x, double y, double p)
	{
		return (double)0;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::thresh_pos>(double x, double y, double p)
	{
		return (double)0;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::thresh_neg>(double x, double y, double p)
	{
		return (double)0;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::step_sym>(double x, double y, double p)
	{
		return x >= (double)1 ? (double)0 : x <= (double)-1 ? (double)0 : p;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::step_pos>(double x, double y, double p)
	{
		return x >= (double)1 ? (double)0 : x <= (double)0 ? (double)0 : p;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::step_neg>(double x, double y, double p)
	{
		return x >= (double)0 ? (double)0 : x <= (double)-1 ? (double)0 : p;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::rad_bas_pos>(double x, double y, double p)
	{
		return (double)2 * x * y / abs(p);
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::rad_bas_neg>(double x, double y, double p)
	{
		return (double)2 * x * y / abs(p);
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::sigmoid_log>(double x, double y, double p)
	{
		return ((double)1 - y) * y * p;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::sigmoid_rat>(double x, double y, double p)
	{
		return p / ((abs(x) + p) * (abs(x) + p));
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::atan>(double x, double y, double p)
	{
		return p / ((double)1 + x * x * p * p);
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::tanh>(double x, double y, double p)
	{
		return ((double)1 - y * y) * p;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::elu>(double x, double y, double p)
	{
		return (x >= (double)0 ? (double)1 : exp(x)) * p;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::gelu>(double x, double y, double p)
	{
		if (x == (double)0)
			return (double)1;
		else
		{
			double z = y / x;
			return ((double)1 + (double)0.797884560802865355 * ((double)1
				+ (double)3 * (double)0.044715 * x * x * p) * ((double)2 - z) * x) * z;
		}
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::lelu>(double x, double y, double p)
	{
		return x >= (double)0 ? (double)1 : p;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::relu>(double x, double y, double p)
	{
		return x >= (double)0 ? p : (double)0;
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::mish>(double x, double y, double p)
	{
		if (x == (double)0)
			return tanh((double)0.693147180559945309 / p);
		else
		{
			double z = y / x;
			return (exp(x * p) * x / ((double)1 + exp(x * p))) * ((double)1 - z * z) + z;
		}
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::swish>(double x, double y, double p)
	{
		if (x == (double)0)
			return (double)0.5;
		else
		{
			double z = y / x;
			return ((double)1 + x * p - y * p) * z;
		}
	}

	template <>
	inline double derivation<fnn_params::fnn_activ::softplus>(double x, double y, double p)
	{
		return exp(x * p) / ((double)1 + exp(x * p));
	}
}