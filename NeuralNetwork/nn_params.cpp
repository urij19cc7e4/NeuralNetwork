#include "nn_params.h"
#include "nn_activs.h"
#include "nn_derivs.h"

using namespace nn_params;

FLT (*nn_params::activs[(uint64_t)nn_activ_t::__count__])(FLT, FLT) =
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

FLT (*nn_params::derivs[(uint64_t)nn_activ_t::__count__])(FLT, FLT) =
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

inline FLT nn_params::activation(FLT x, FLT sx, FLT sy, FLT sz, nn_activ_t activ)
{
	return activs[(uint64_t)activ](x * sx, sz) * sy;
}

inline FLT nn_params::derivation(FLT x, FLT sx, FLT sy, FLT sz, nn_activ_t activ)
{
	return derivs[(uint64_t)activ](x * sx, sz) * sx * sy;
}