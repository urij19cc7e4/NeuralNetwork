#include "cnn_2_fnn.h"

using namespace std;
using namespace arithmetic;
using namespace nn_params;

cnn_2_fnn::cnn_2_fnn(bool flatten, bool max_pool) noexcept : _flatten(flatten), _max_pool(max_pool) {}

cnn_2_fnn::cnn_2_fnn(ifstream& file) noexcept : cnn_2_fnn()
{
	uint64_t flatten;
	uint64_t max_pool;

	file.read(reinterpret_cast<char*>(&flatten), sizeof(flatten));
	file.read(reinterpret_cast<char*>(&max_pool), sizeof(max_pool));

	_flatten = (bool)flatten;
	_max_pool = (bool)max_pool;
}

cnn_2_fnn::cnn_2_fnn(const cnn_2_fnn_info& i) noexcept : cnn_2_fnn(i.flatten, i.max_pool) {}

cnn_2_fnn::cnn_2_fnn(const cnn_2_fnn& o) noexcept : _flatten(o._flatten), _max_pool(o._max_pool) {}

cnn_2_fnn::cnn_2_fnn(cnn_2_fnn&& o) noexcept : _flatten(o._flatten), _max_pool(o._max_pool) {}

cnn_2_fnn::~cnn_2_fnn() {}

nn* cnn_2_fnn::create_new() const
{
	return (nn*)(new cnn_2_fnn(*this));
}

void cnn_2_fnn::save_to_file(ofstream& file) const
{
	const uint64_t id = (uint64_t)adapter_id::CNN_2_FNN;
	file.write(reinterpret_cast<const char*>(&id), sizeof(id));

	uint64_t flatten = (uint64_t)get_flatten_enabled();
	uint64_t max_pool = (uint64_t)get_max_pool_enabled();

	file.write(reinterpret_cast<const char*>(&flatten), sizeof(flatten));
	file.write(reinterpret_cast<const char*>(&max_pool), sizeof(max_pool));
}

inline bool cnn_2_fnn::get_flatten_enabled() const noexcept
{
	return _flatten;
}

inline bool cnn_2_fnn::get_max_pool_enabled() const noexcept
{
	return _max_pool;
}

inline bool cnn_2_fnn::is_empty() const noexcept
{
	return false;
}

uint64_t cnn_2_fnn::get_param_count() const noexcept
{
	return (uint64_t)0;
}

nn_trainy* cnn_2_fnn::get_trainy(const ::data<FLT>& _data_prev, double _drop_out, bool _delta_hold) const
{
	throw exception("Adapter must be between layers.");
}

nn_trainy* cnn_2_fnn::get_trainy(const nn_trainy& _data_prev, double _drop_out, bool _delta_hold) const
{
	const tns<FLT>& cnn_data = (const tns<FLT>&)(((const cnn_trainy&)_data_prev)._activ);
	return (nn_trainy*)(new cnn_2_fnn_trainy(cnn_data, _max_pool, _flatten, _drop_out));
}

nn_trainy_batch* cnn_2_fnn::get_trainy_batch(const ::data<FLT>& _data_prev) const
{
	throw exception("Adapter must be between layers.");
}

nn_trainy_batch* cnn_2_fnn::get_trainy_batch(const nn_trainy& _data_prev) const
{
	const tns<FLT>& cnn_data = (const tns<FLT>&)(((const cnn_trainy&)_data_prev)._activ);
	return (nn_trainy_batch*)(new cnn_2_fnn_trainy_batch(cnn_data));
}

::data<FLT>* cnn_2_fnn::pass_fwd(const ::data<FLT>& _data) const
{
	if (_flatten)
		return (::data<FLT>*)(new vec<FLT>(move(tns<FLT>((const tns<FLT>&)_data).to_vec())));
	else
		return (::data<FLT>*)(new vec<FLT>(move(pool_full((const tns<FLT>&)_data, _max_pool))));
}

FLT cnn_2_fnn::train_bwd(nn_trainy& _data, const ::data<FLT>& _data_next) const
{
	throw exception("Adapter must be between layers.");
}

void cnn_2_fnn::train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const
{
	const cnn_2_fnn_trainy& adp_data = (const cnn_2_fnn_trainy&)_data;
	cnn_trainy& cnn_data = (cnn_trainy&)_data_prev;

	if (_flatten)
	{
		uint64_t flat_scale;
		cnn_data._bias_gd = (FLT)0;

		if (cnn_data._pool)
		{
			flat_scale = cnn_data._pool_temp_2.get_size_2() * cnn_data._pool_temp_2.get_size_3();

			for (uint64_t i = (uint64_t)0; i < cnn_data._pool_temp_2.get_size(); ++i)
				cnn_data._pool_temp_2(i) = adp_data._link_gd(i);

			pool_bwd(cnn_data._pool_temp_2, cnn_data._pool_map, cnn_data._core_gd);
		}
		else
		{
			flat_scale = cnn_data._core_gd.get_size_2() * cnn_data._core_gd.get_size_3();

			for (uint64_t i = (uint64_t)0; i < cnn_data._core_gd.get_size(); ++i)
				cnn_data._core_gd(i) = adp_data._link_gd(i);
		}

		for (uint64_t i = (uint64_t)0; i < adp_data._bias_gd.get_size(); ++i)
			cnn_data._bias_gd(i / flat_scale) += adp_data._bias_gd(i);
	}
	else
	{
		if (cnn_data._pool)
		{
			pool_full_bwd(adp_data._link_gd, adp_data._pool_map, cnn_data._pool_temp_2, adp_data._max_pool);
			pool_bwd(cnn_data._pool_temp_2, cnn_data._pool_map, cnn_data._core_gd);
		}
		else
			pool_full_bwd(adp_data._link_gd, adp_data._pool_map, cnn_data._core_gd, adp_data._max_pool);

		cnn_data._bias_gd = adp_data._bias_gd;
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
	const cnn_trainy& cnn_data = (const cnn_trainy&)_data_prev;

	if (_flatten)
	{
		uint64_t layer_size = adp_data._drop_map.get_size() / cnn_data._drop_map.get_size();

		for (uint64_t i = (uint64_t)0; i < adp_data._activ.get_size(); ++i)
		{
			adp_data._drop_map(i) = cnn_data._drop_map(i / layer_size);

			adp_data._activ(i) = cnn_data._activ(i);
			adp_data._deriv(i) = adp_data._drop_map(i) ? (FLT)1 : (FLT)0;
		}
	}
	else
	{
		adp_data._drop_map = cnn_data._drop_map;
		pool_full_fwd(cnn_data._activ, adp_data._pool_map, adp_data._activ, _max_pool);

		for (uint64_t i = (uint64_t)0; i < adp_data._deriv.get_size(); ++i)
			adp_data._deriv(i) = adp_data._drop_map(i) ? (FLT)1 : (FLT)0;
	}
}

void cnn_2_fnn::train_upd(const nn_trainy& _data) {}

void cnn_2_fnn::train_upd(const nn_trainy_batch& _data) {}

cnn_2_fnn& cnn_2_fnn::operator=(const cnn_2_fnn& o) noexcept
{
	_flatten = o._flatten;
	_max_pool = o._max_pool;

	return *this;
}

cnn_2_fnn& cnn_2_fnn::operator=(cnn_2_fnn&& o) noexcept
{
	_flatten = o._flatten;
	_max_pool = o._max_pool;

	return *this;
}

cnn_2_fnn_trainy::cnn_2_fnn_trainy(const tns<FLT>& data, bool max_pool, bool flatten, double drop_out)
	: fnn_trainy((uint64_t)0, flatten ? data.get_size() : data.get_size_1(), drop_out, false),
	_pool_map(max_pool ? data.get_size_1() : (uint64_t)0), _flatten(flatten), _max_pool(max_pool) {}

cnn_2_fnn_trainy::~cnn_2_fnn_trainy() {}

void cnn_2_fnn_trainy::update(const ::data<FLT>& _data_prev, FLT alpha, FLT speed) {}

void cnn_2_fnn_trainy::update(const nn_trainy& _data_prev, FLT alpha, FLT speed) {}

cnn_2_fnn_trainy_batch::cnn_2_fnn_trainy_batch(const tns<FLT>& data) : fnn_trainy_batch((uint64_t)0, (uint64_t)0) {}

cnn_2_fnn_trainy_batch::~cnn_2_fnn_trainy_batch() {}

void cnn_2_fnn_trainy_batch::begin_update(FLT alpha) {}

void cnn_2_fnn_trainy_batch::update(const nn_trainy& _data, const ::data<FLT>& _data_prev, FLT speed) {}

void cnn_2_fnn_trainy_batch::update(const nn_trainy& _data, const nn_trainy& _data_prev, FLT speed) {}

cnn_2_fnn_info::cnn_2_fnn_info(bool flatten, bool max_pool) : flatten(flatten), max_pool(max_pool) {}

nn* cnn_2_fnn_info::create_new() const
{
	return (nn*)(new cnn_2_fnn(*this));
}