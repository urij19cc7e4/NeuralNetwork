#include "layer_adapters.h"

using namespace std;
using namespace arithmetic;
using namespace nn_params;

cnn_2_fnn::cnn_2_fnn(bool max_pool) noexcept : _max_pool(max_pool) {}

cnn_2_fnn::cnn_2_fnn(const cnn_2_fnn_info&i) noexcept : cnn_2_fnn(i.max_pool) {}

cnn_2_fnn::cnn_2_fnn(const cnn_2_fnn& o) noexcept : _max_pool(o._max_pool) {}

cnn_2_fnn::cnn_2_fnn(cnn_2_fnn&& o) noexcept : _max_pool(o._max_pool) {}

cnn_2_fnn::~cnn_2_fnn() {}

nn*cnn_2_fnn::create_new() const
{
	return (nn*)(new cnn_2_fnn(*this));
}

uint64_t cnn_2_fnn::get_param_count() const noexcept
{
	return (uint64_t)0;
}

nn_trainy* cnn_2_fnn::get_trainy(const ::data<FLT>& _data_prev, bool _drop_out) const
{
	throw exception("Adapter must be between layers.");
}

nn_trainy* cnn_2_fnn::get_trainy(const nn_trainy& _data_prev, bool _drop_out) const
{
	const tns<FLT>& cnn_data = (const tns<FLT>&)(((const cnn_trainy&)_data_prev)._activ);
	return (nn_trainy*)(new cnn_2_fnn_trainy(cnn_data, _max_pool, _flatten, _drop_out));
}

::data<FLT>*cnn_2_fnn::pass_fwd(const ::data<FLT>&_data) const
{
	if(_flatten)
		return (::data<FLT>*)(new vec<FLT>(move(tns<FLT>((const tns<FLT>&)_data).to_vec())));
	else
	return (::data<FLT>*)(new vec<FLT>(move(pool_full((const tns<FLT>&)_data,_max_pool))));
}

FLT cnn_2_fnn::train_bwd(nn_trainy& _data, const ::data<FLT>& _data_next) const
{
	throw exception("Adapter must be between layers.");
}

void cnn_2_fnn::train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const
{
	const cnn_2_fnn_trainy& adp_data = (const cnn_2_fnn_trainy&)_data;
	cnn_trainy& cnn_data = (cnn_trainy&)_data_prev;
	const fnn_trainy& fnn_data = (const fnn_trainy&)_data;

	if (_flatten)
	{
		uint64_t flat_size;
		cnn_data._bias_gd = (FLT)0;

		if (cnn_data._pool)
		{
			flat_size = cnn_data._pool_temp_2.get_size_2() * cnn_data._pool_temp_2.get_size_3();

			for (uint64_t i = (uint64_t)0; i < cnn_data._pool_temp_2.get_size(); ++i)
				cnn_data._pool_temp_2(i) = adp_data._link_gd(i);

			pool_bwd(cnn_data._pool_temp_2, cnn_data._pool_map, cnn_data._core_gd);
		}
		else
		{
			flat_size = cnn_data._core_gd.get_size_2() * cnn_data._core_gd.get_size_3();

			for (uint64_t i = (uint64_t)0; i < cnn_data._core_gd.get_size(); ++i)
				cnn_data._core_gd(i) = fnn_data._link_gd(i);
		}

		for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < cnn_data._bias_gd.get_size(); ++i)
			for (uint64_t k = (uint64_t)0; k < flat_size; ++j, ++k)
				cnn_data._bias_gd(i) += fnn_data._bias_gd(j);
	}
	else
	{
		if (cnn_data._pool)
		{
			pool_full_bwd(fnn_data._link_gd, adp_data._pool_map,
				cnn_data._pool_temp_2, adp_data._max_pool);
			pool_bwd(cnn_data._pool_temp_2, cnn_data._pool_map, cnn_data._core_gd);
		}
		else
		{
			pool_full_bwd(fnn_data._link_gd, adp_data._pool_map,
				cnn_data._core_gd, adp_data._max_pool);
		}

		cnn_data._bias_gd = fnn_data._bias_gd;
	}

	cnn_data._core_gd *= cnn_data._deriv;
}

void cnn_2_fnn::train_fwd(nn_trainy& _data, const ::data<FLT>& _data_prev) const
{
	throw exception("Adapter must be between layers.");
}

void cnn_2_fnn::train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const
{
	cnn_2_fnn_trainy& adp_data = (cnn_2_fnn_trainy&)_data;
	fnn_trainy& fnn_data = (fnn_trainy&)_data;

	if(_flatten)
		for(uint64_t i=(uint64_t)0;i<fnn_data._activ.get_size();++i)
		fnn_data._activ(i) = (((const cnn_trainy&)_data_prev)._activ)(i);
	else
	pool_full_fwd(((const cnn_trainy&)_data_prev)._activ, adp_data._pool_map, fnn_data._activ, adp_data._max_pool);
}

void cnn_2_fnn::train_upd(const nn_trainy& _data) {}

cnn_2_fnn& cnn_2_fnn::operator=(const cnn_2_fnn& o) noexcept
{
	_max_pool = o._max_pool;

	return *this;
}

cnn_2_fnn& cnn_2_fnn::operator=(cnn_2_fnn&& o) noexcept
{
	_max_pool = o._max_pool;

	return *this;
}

cnn_2_fnn_trainy::cnn_2_fnn_trainy(const tns<FLT>& data, bool max_pool, bool flatten, bool drop_out)
	: fnn_trainy((uint64_t)0, flatten?data.get_size():data.get_size_1(), drop_out),
	_pool_map(max_pool ? data.get_size_1() : (uint64_t)0),
	_flatten(flatten), _max_pool(max_pool)
{
	_deriv = (FLT)1;
}

cnn_2_fnn_trainy::~cnn_2_fnn_trainy() {}

void cnn_2_fnn_trainy::update(const ::data<FLT>& _data_prev, FLT alpha, FLT speed) {}

void cnn_2_fnn_trainy::update(const nn_trainy& _data_prev, FLT alpha, FLT speed) {}