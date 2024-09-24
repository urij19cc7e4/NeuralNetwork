#include "nn_params.h"
#include "nn_activs.h"
#include "nn_derivs.h"

using namespace nn_params;

nn_params::nn_activ_params::nn_activ_params
(
	nn_activ_t activ_type,
	FLT scale_x,
	FLT scale_y,
	FLT shift_x,
	FLT shift_y
) noexcept :
	activ_type(activ_type),
	scale_x(scale_x),
	scale_y(scale_y),
	shift_x(shift_x),
	shift_y(shift_y)
{}

FLT (*nn_params::activs[(uint64_t)nn_activ_t::__count__])(FLT) =
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

FLT (*nn_params::derivs[(uint64_t)nn_activ_t::__count__])(FLT) =
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

inline FLT nn_params::activation(FLT x, const nn_activ_params& p)
{
	return activs[(uint64_t)p.activ_type](x / p.scale_x - p.shift_x)
		* p.scale_y + p.shift_y;
}

inline FLT nn_params::derivation(FLT x, const nn_activ_params& p)
{
	return derivs[(uint64_t)p.activ_type](x / p.scale_x - p.shift_x)
		* p.scale_y / p.scale_x;
}