#pragma once

#include <cmath>
#include <random>

#include "nn_params.h"

namespace nn_inits
{
	template <nn_params::nn_activ_t activ, nn_params::nn_init_t init>
	inline nn_params::nn_init<init> initialization(uint64_t isize, uint64_t osize, double param);

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::signed_pos, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::signed_pos, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::signed_neg, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::signed_neg, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::thresh_pos, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::thresh_pos, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::thresh_neg, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::thresh_neg, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::step_sym, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::step_sym, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::step_pos, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::step_pos, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::step_neg, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::step_neg, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::rad_bas_pos, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::rad_bas_pos, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::rad_bas_neg, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::rad_bas_neg, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::sigmoid_log, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::sigmoid_log, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::sigmoid_rat, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::sigmoid_rat, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::atan, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::atan, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::tanh, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::tanh, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::elu, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::elu, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::gelu, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::gelu, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::lelu, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / ((double)isize * ((double)1 + param * param)))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::lelu, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / ((double)isize * ((double)1 + param * param))),
			(double)-1 * sqrt((double)6 / ((double)isize * ((double)1 + param * param)))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::relu, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::relu, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::mish, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::mish, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::swish, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::swish, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)isize),
			(double)-1 * sqrt((double)6 / (double)isize)
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::normal>
		initialization<nn_params::nn_activ_t::softplus, nn_params::nn_init_t::normal>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::normal>
		{
			(double)0,
			sqrt((double)2 / (double)(isize + osize))
		};
	}

	template <>
	inline nn_params::nn_init<nn_params::nn_init_t::uniform>
		initialization<nn_params::nn_activ_t::softplus, nn_params::nn_init_t::uniform>(
			uint64_t isize, uint64_t osize, double param)
	{
		return nn_params::nn_init<nn_params::nn_init_t::uniform>
		{
			sqrt((double)6 / (double)(isize + osize)),
			(double)-1 * sqrt((double)6 / (double)(isize + osize))
		};
	}

	template <nn_params::nn_activ_t activ, nn_params::nn_init_t init>
	class initializer;

	template <nn_params::nn_activ_t activ>
	class initializer<activ, nn_params::nn_init_t::normal>
	{
	private:
		std::normal_distribution<double> distributor;

	public:
		initializer() = delete;

		initializer(uint64_t isize, uint64_t osize, double param = (double)1)
		{
			nn_params::nn_init<nn_params::nn_init_t::normal> params =
				initialization<activ, nn_params::nn_init_t::normal>(isize, osize, param);

			distributor = std::normal_distribution<double>(params.mean, params.sigm);
		}

		initializer(const initializer& o) = delete;

		initializer(initializer&& o) = delete;

		~initializer() {}

		template <typename URBG>
		double operator()(URBG& engine)
		{
			return distributor(engine);
		}
	};

	template <nn_params::nn_activ_t activ>
	class initializer<activ, nn_params::nn_init_t::uniform>
	{
	private:
		std::uniform_real_distribution<double> distributor;

	public:
		initializer() = delete;

		initializer(uint64_t isize, uint64_t osize, double param = (double)1)
		{
			nn_params::nn_init<nn_params::nn_init_t::uniform> params =
				initialization<activ, nn_params::nn_init_t::uniform>(isize, osize, param);

			distributor = std::uniform_real_distribution<double>(params.min, params.max);
		}

		initializer(const initializer& o) = delete;

		initializer(initializer&& o) = delete;

		~initializer() {}

		template <typename URBG>
		double operator()(URBG& engine)
		{
			return distributor(engine);
		}
	};
}