#include "nn_params.h"
#include "nn_activs.h"
#include "nn_derivs.h"
#include "nn_inits.h"

using namespace nn_params;

double (*nn_params::activs[(uint64_t)nn_activ_t::__count__])(double, double) =
{
	nn_activs::activation<nn_activ_t::signed_pos>,
	nn_activs::activation<nn_activ_t::signed_neg>,
	nn_activs::activation<nn_activ_t::thresh_pos>,
	nn_activs::activation<nn_activ_t::thresh_neg>,
	nn_activs::activation<nn_activ_t::step_sym>,
	nn_activs::activation<nn_activ_t::step_pos>,
	nn_activs::activation<nn_activ_t::step_neg>,
	nn_activs::activation<nn_activ_t::rad_bas_pos>,
	nn_activs::activation<nn_activ_t::rad_bas_neg>,
	nn_activs::activation<nn_activ_t::sigmoid_log>,
	nn_activs::activation<nn_activ_t::sigmoid_rat>,
	nn_activs::activation<nn_activ_t::atan>,
	nn_activs::activation<nn_activ_t::tanh>,
	nn_activs::activation<nn_activ_t::elu>,
	nn_activs::activation<nn_activ_t::gelu>,
	nn_activs::activation<nn_activ_t::lelu>,
	nn_activs::activation<nn_activ_t::relu>,
	nn_activs::activation<nn_activ_t::mish>,
	nn_activs::activation<nn_activ_t::swish>,
	nn_activs::activation<nn_activ_t::softplus>
};

double (*nn_params::derivs[(uint64_t)nn_activ_t::__count__])(double, double, double) =
{
	nn_derivs::derivation<nn_activ_t::signed_pos>,
	nn_derivs::derivation<nn_activ_t::signed_neg>,
	nn_derivs::derivation<nn_activ_t::thresh_pos>,
	nn_derivs::derivation<nn_activ_t::thresh_neg>,
	nn_derivs::derivation<nn_activ_t::step_sym>,
	nn_derivs::derivation<nn_activ_t::step_pos>,
	nn_derivs::derivation<nn_activ_t::step_neg>,
	nn_derivs::derivation<nn_activ_t::rad_bas_pos>,
	nn_derivs::derivation<nn_activ_t::rad_bas_neg>,
	nn_derivs::derivation<nn_activ_t::sigmoid_log>,
	nn_derivs::derivation<nn_activ_t::sigmoid_rat>,
	nn_derivs::derivation<nn_activ_t::atan>,
	nn_derivs::derivation<nn_activ_t::tanh>,
	nn_derivs::derivation<nn_activ_t::elu>,
	nn_derivs::derivation<nn_activ_t::gelu>,
	nn_derivs::derivation<nn_activ_t::lelu>,
	nn_derivs::derivation<nn_activ_t::relu>,
	nn_derivs::derivation<nn_activ_t::mish>,
	nn_derivs::derivation<nn_activ_t::swish>,
	nn_derivs::derivation<nn_activ_t::softplus>
};

nn_init<nn_init_t::normal> (*nn_params::inits_normal[(uint64_t)nn_activ_t::__count__])(uint64_t, uint64_t, double) =
{
	nn_inits::initialization<nn_activ_t::signed_pos, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::signed_neg, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::thresh_pos, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::thresh_neg, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::step_sym, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::step_pos, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::step_neg, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::rad_bas_pos, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::rad_bas_neg, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::sigmoid_log, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::sigmoid_rat, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::atan, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::tanh, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::elu, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::gelu, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::lelu, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::relu, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::mish, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::swish, nn_init_t::normal>,
	nn_inits::initialization<nn_activ_t::softplus, nn_init_t::normal>
};

nn_init<nn_init_t::uniform> (*nn_params::inits_uniform[(uint64_t)nn_activ_t::__count__])(uint64_t, uint64_t, double) =
{
	nn_inits::initialization<nn_activ_t::signed_pos, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::signed_neg, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::thresh_pos, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::thresh_neg, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::step_sym, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::step_pos, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::step_neg, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::rad_bas_pos, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::rad_bas_neg, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::sigmoid_log, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::sigmoid_rat, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::atan, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::tanh, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::elu, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::gelu, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::lelu, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::relu, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::mish, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::swish, nn_init_t::uniform>,
	nn_inits::initialization<nn_activ_t::softplus, nn_init_t::uniform>
};

inline double nn_params::activation(double x, double p, nn_activ_t activ)
{
	return activs[(uint64_t)activ](x, p);
}

inline double nn_params::derivation(double x, double y, double p, nn_activ_t activ)
{
	return derivs[(uint64_t)activ](x, y, p);
}

inline nn_init<nn_init_t::normal> nn_params::init_normal(uint64_t isize, uint64_t osize, double param, nn_activ_t activ)
{
	return inits_normal[(uint64_t)activ](isize, osize, param);
}

inline nn_init<nn_init_t::uniform> nn_params::init_uniform(uint64_t isize, uint64_t osize, double param, nn_activ_t activ)
{
	return inits_uniform[(uint64_t)activ](isize, osize, param);
}