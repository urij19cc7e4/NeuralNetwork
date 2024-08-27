#include "cnn.h"

#include "rand_init.h"

using namespace std;
using namespace arithmetic;
using namespace nn_params;

cnn::cnn() noexcept
	: _core(), _bias(), _bn_link((FLT)1), _bn_bias((FLT)0), _activ(nn_activ_t::__count__),
	_scale_x((FLT)1), _scale_y((FLT)1), _scale_z((FLT)1), _convo(nn_convo_t::__count__), _pool(false) {}

cnn::cnn(uint64_t height, uint64_t width, uint64_t size, uint64_t next_height, uint64_t next_width,
	nn_activ_t activ, nn_init_t init, FLT scale_x, FLT scale_y, FLT scale_z, nn_convo_t convo, bool pool)
	: _core(size, height, width), _bias(size), _bn_link((FLT)1), _bn_bias((FLT)0), _activ(activ),
	_scale_x(scale_x), _scale_y(scale_y), _scale_z(scale_z), _convo(convo), _pool(pool)
{
	if (height == (uint64_t)0 || width == (uint64_t)0 || size == (uint64_t)0 || scale_x == (FLT)0 || scale_y == (FLT)0 || scale_z == (FLT)0
		|| convo >= nn_convo_t::__count__ || activ >= nn_activ_t::__count__ || init >= nn_init_t::__count__)
		throw exception(error_msg::cnn_wrong_init_error);
	else
	{
		if (next_height == (uint64_t)0 || next_width == (uint64_t)0)
		{
			next_height = height;
			next_width = width;
		}

		rand_init_base* randomizer = nullptr;

		__assume(init < nn_init_t::__count__);
		switch (init)
		{
		case nn_init_t::normal:
			randomizer = new rand_init<nn_init_t::normal>(height * width, next_height * next_width);
			break;

		case nn_init_t::uniform:
			randomizer = new rand_init<nn_init_t::uniform>(height * width, next_height * next_width);
			break;

		default:
			__assume(false);
		}

		for (uint64_t i = (uint64_t)0; i < _core.get_size(); ++i)
			_core(i) = (*randomizer)();

		_bias = (FLT)0;

		delete randomizer;
	}
}

cnn::cnn(const tns<FLT>& core, const vec<FLT>& bias, FLT bn_link, FLT bn_bias,
	nn_activ_t activ, FLT scale_x, FLT scale_y, FLT scale_z, nn_convo_t convo, bool pool)
	: _core(core), _bias(bias), _bn_link(bn_link), _bn_bias(bn_bias), _activ(activ),
	_scale_x(scale_x), _scale_y(scale_y), _scale_z(scale_z), _convo(convo), _pool(pool) {}

cnn::cnn(tns<FLT>&& core, vec<FLT>&& bias, FLT bn_link, FLT bn_bias,
	nn_activ_t activ, FLT scale_x, FLT scale_y, FLT scale_z, nn_convo_t convo, bool pool) noexcept
	: _core(move(core)), _bias(move(bias)), _bn_link(bn_link), _bn_bias(bn_bias), _activ(activ),
	_scale_x(scale_x), _scale_y(scale_y), _scale_z(scale_z), _convo(convo), _pool(pool) {}

cnn::cnn(const cnn& o)
	: _core(o._core), _bias(o._bias), _bn_link(o._bn_link), _bn_bias(o._bn_bias), _activ(o._activ),
	_scale_x(o._scale_x), _scale_y(o._scale_y), _scale_z(o._scale_z), _convo(o._convo), _pool(o._pool) {}

cnn::cnn(cnn&& o) noexcept
	: _core(move(o._core)), _bias(move(o._bias)), _bn_link(o._bn_link), _bn_bias(o._bn_bias), _activ(o._activ),
	_scale_x(o._scale_x), _scale_y(o._scale_y), _scale_z(o._scale_z), _convo(o._convo), _pool(o._pool) {}

cnn::~cnn() {}

inline nn_activ_t cnn::get_activ_type() const noexcept
{
	return _activ;
}

inline nn_convo_t cnn::get_convo_type() const noexcept
{
	return _convo;
}

inline FLT cnn::get_scale_x() const noexcept
{
	return _scale_x;
}

inline FLT cnn::get_scale_y() const noexcept
{
	return _scale_y;
}

inline FLT cnn::get_scale_z() const noexcept
{
	return _scale_z;
}

inline FLT cnn::get_bn_link() const noexcept
{
	return _bn_link;
}

inline FLT cnn::get_bn_bias() const noexcept
{
	return _bn_bias;
}

inline bool cnn::get_pool_enabled() const noexcept
{
	return _pool;
}

inline uint64_t cnn::get_height() const noexcept
{
	return _core.get_size_2();
}

inline uint64_t cnn::get_width() const noexcept
{
	return _core.get_size_3();
}

inline uint64_t cnn::get_size() const noexcept
{
	return _core.get_size_1();
}

inline const tns<FLT>& cnn::get_core() const noexcept
{
	return _core;
}

inline const vec<FLT>& cnn::get_bias() const noexcept
{
	return _bias;
}

inline bool cnn::is_empty() const noexcept
{
	return _core.is_empty() || _bias.is_empty();
}

uint64_t cnn::get_param_count() const noexcept
{
	return _core.get_size() + _bias.get_size();
}

nn_trainy* cnn::get_trainy(const data<FLT>& _data_prev) const
{
	const tns<FLT>& cnn_data = (const tns<FLT>&)_data_prev;
	return new cnn_trainy(cnn_data, _core, _convo, _pool);
}

nn_trainy* cnn::get_trainy(const nn_trainy& _data_prev) const
{
	const tns<FLT>& cnn_data = (const tns<FLT>&)(((const cnn_trainy&)_data_prev)._activ);
	return new cnn_trainy(cnn_data, _core, _convo, _pool);
}

void cnn::pass_fwd(data<FLT>& _data) const
{
	if (is_empty())
		throw exception(error_msg::cnn_empty_error);
	else
	{
		tns<FLT>& result = (tns<FLT>&)_data;
		result = move(convolute(result, _core, _convo));
		uint64_t layer_size = result.get_size_2() * result.get_size_3();

		__assume(result.get_size_1() == _bias.get_size());
		for (uint64_t i = (uint64_t)0; i < result.get_size(); ++i)
			result(i) = activation(result(i) + _bias(i / layer_size), _scale_x, _scale_y, _scale_z, _activ);

		if (_pool)
			result = move(pool(result));
	}
}

FLT cnn::train_bwd(nn_trainy& _data, const data<FLT>& _data_next) const
{
	if (is_empty())
		throw exception(error_msg::cnn_empty_error);
	else
	{
		FLT error(0), bias_gd(0);
		cnn_trainy& cnn_data = (cnn_trainy&)_data;
		const tns<FLT>& res_data = (const tns<FLT>&)_data_next;
		uint64_t layer_size = res_data.get_size_2() * res_data.get_size_3();

		if (cnn_data._pool)
		{
			__assume(cnn_data._pool_temp_2.get_size() == res_data.get_size());
			__assume(cnn_data._pool_temp_2.get_size() == cnn_data._activ.get_size());
			for (uint64_t i = (uint64_t)0; i < cnn_data._pool_temp_2.get_size(); ++i)
			{
				FLT loss = res_data(i) - cnn_data._activ(i);

				cnn_data._pool_temp_2(i) = loss;
				error += loss * loss;
			}

			pool_bwd(cnn_data._pool_temp_2, cnn_data._pool_map, cnn_data._core_gd);

			collapse(cnn_data._core_gd, cnn_data._bias_gd);
			cnn_data._core_gd *= cnn_data._deriv;
		}
		else
		{
			cnn_data._bias_gd = FLT(0);

			__assume(cnn_data._core_gd.get_size() == res_data.get_size());
			__assume(cnn_data._core_gd.get_size() == cnn_data._activ.get_size());
			__assume(cnn_data._core_gd.get_size() == cnn_data._deriv.get_size());
			__assume(cnn_data._core_gd.get_size_1() == cnn_data._bias_gd.get_size());
			for (uint64_t i = (uint64_t)0; i < cnn_data._core_gd.get_size(); ++i)
			{
				FLT loss = res_data(i) - cnn_data._activ(i);

				cnn_data._core_gd(i) = cnn_data._deriv(i) * loss;
				cnn_data._bias_gd(i / layer_size) += loss;
				error += loss * loss;
			}
		}

		return error;
	}
}

void cnn::train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const
{
	if (is_empty())
		throw exception(error_msg::cnn_empty_error);
	else
	{
		cnn_trainy& cnn_data = (cnn_trainy&)_data_prev;
		tns<FLT> core_rot(rotate(_core));

		if (cnn_data._pool)
		{
			convolute_bwd(((const cnn_trainy&)_data)._core_gd, core_rot, cnn_data._pool_temp_2, _convo);
			pool_bwd(cnn_data._pool_temp_2, cnn_data._pool_map, cnn_data._core_gd);
		}
		else
			convolute_bwd(((const cnn_trainy&)_data)._core_gd, core_rot, cnn_data._core_gd, _convo);

		collapse(cnn_data._core_gd, cnn_data._bias_gd);
		cnn_data._core_gd *= cnn_data._deriv;
	}
}

void cnn::train_fwd(nn_trainy& _data, const data<FLT>& _data_prev) const
{
	if (is_empty())
		throw exception(error_msg::cnn_empty_error);
	else
	{
		cnn_trainy& cnn_data = (cnn_trainy&)_data;

		if (cnn_data._pool)
		{
			convolute_fwd((const tns<FLT>&)_data_prev, _core, cnn_data._pool_temp_1, _convo);
			uint64_t layer_size = cnn_data._pool_temp_1.get_size_2() * cnn_data._pool_temp_1.get_size_3();

			__assume(cnn_data._pool_temp_1.get_size_1() == _bias.get_size());
			__assume(cnn_data._deriv.get_size_1() == _bias.get_size());
			for (uint64_t i = (uint64_t)0; i < cnn_data._pool_temp_1.get_size(); ++i)
			{
				FLT value = cnn_data._pool_temp_1(i) + _bias(i / layer_size);

				cnn_data._pool_temp_1(i) = activation(value, _scale_x, _scale_y, _scale_z, _activ);
				cnn_data._deriv(i) = derivation(value, _scale_x, _scale_y, _scale_z, _activ);
			}

			pool_fwd(cnn_data._pool_temp_1, cnn_data._pool_map, cnn_data._activ);
		}
		else
		{
			convolute_fwd((const tns<FLT>&)_data_prev, _core, cnn_data._activ, _convo);
			uint64_t layer_size = cnn_data._activ.get_size_2() * cnn_data._activ.get_size_3();

			__assume(cnn_data._activ.get_size_1() == _bias.get_size());
			__assume(cnn_data._deriv.get_size_1() == _bias.get_size());
			__assume(cnn_data._activ.get_size() == cnn_data._deriv.get_size());
			for (uint64_t i = (uint64_t)0; i < cnn_data._activ.get_size(); ++i)
			{
				FLT value = cnn_data._activ(i) + _bias(i / layer_size);

				cnn_data._activ(i) = activation(value, _scale_x, _scale_y, _scale_z, _activ);
				cnn_data._deriv(i) = derivation(value, _scale_x, _scale_y, _scale_z, _activ);
			}
		}
	}
}

void cnn::train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const
{
	train_fwd(_data, ((const cnn_trainy&)_data_prev)._activ);
}

void cnn::train_upd(const nn_trainy& _data)
{
	if (is_empty())
		throw exception(error_msg::cnn_empty_error);
	else
	{
		const cnn_trainy& cnn_data = (const cnn_trainy&)_data;

		_core += cnn_data._core_dt;
		_bias += cnn_data._bias_dt;

		_bn_link += cnn_data._bn_link_dt;
		_bn_bias += cnn_data._bn_bias_dt;
	}
}

cnn& cnn::operator=(const cnn& o)
{
	_core = o._core;
	_bias = o._bias;
	_bn_link = o._bn_link;
	_bn_bias = o._bn_bias;
	_activ = o._activ;
	_scale_x = o._scale_x;
	_scale_y = o._scale_y;
	_scale_z = o._scale_z;
	_convo = o._convo;
	_pool = o._pool;

	return *this;
}

cnn& cnn::operator=(cnn&& o) noexcept
{
	_core = move(o._core);
	_bias = move(o._bias);
	_bn_link = o._bn_link;
	_bn_bias = o._bn_bias;
	_activ = o._activ;
	_scale_x = o._scale_x;
	_scale_y = o._scale_y;
	_scale_z = o._scale_z;
	_convo = o._convo;
	_pool = o._pool;

	return *this;
}

cnn_trainy::cnn_trainy(const tns<FLT>& data, const tns<FLT>& core, nn_convo_t convo, bool pool)
	: _activ(), _deriv(), _core_gd(), _bias_gd(), _pool_map(), _pool_temp_1(), _pool_temp_2(),
	_core_dt(core.get_size_1(), core.get_size_2(), core.get_size_3()),
	_core_dt_temp(core.get_size_1(), core.get_size_2(), core.get_size_3()),
	_bias_dt(core.get_size_1()), _bias_dt_temp(core.get_size_1()),
	_bn_link_dt((FLT)0), _bn_bias_dt((FLT)0), _bn_link_gd((FLT)0), _bn_bias_gd((FLT)0),
	_convo(convo), _pool(pool)
{
	if (pool)
	{
		tns<FLT> convoluted(move(convolute(data, core, convo)));
		tns<FLT> pooled(move(arithmetic::pool(convoluted)));

		_activ = move(tns<FLT>(pooled.get_size_1(), pooled.get_size_2(), pooled.get_size_3()));
		_deriv = move(tns<FLT>(convoluted.get_size_1(), convoluted.get_size_2(), convoluted.get_size_3()));
		_core_gd = move(tns<FLT>(convoluted.get_size_1(), convoluted.get_size_2(), convoluted.get_size_3()));
		_bias_gd = move(vec<FLT>(convoluted.get_size_1()));

		_pool_map = move(tns<uint8_t>(pooled.get_size_1(), pooled.get_size_2(), pooled.get_size_3()));
		_pool_temp_1 = move(tns<FLT>(convoluted.get_size_1(), convoluted.get_size_2(), convoluted.get_size_3()));
		_pool_temp_2 = move(tns<FLT>(pooled.get_size_1(), pooled.get_size_2(), pooled.get_size_3()));
	}
	else
	{
		tns<FLT> convoluted(move(convolute(data, core, convo)));

		_activ = move(tns<FLT>(convoluted.get_size_1(), convoluted.get_size_2(), convoluted.get_size_3()));
		_deriv = move(tns<FLT>(convoluted.get_size_1(), convoluted.get_size_2(), convoluted.get_size_3()));
		_core_gd = move(tns<FLT>(convoluted.get_size_1(), convoluted.get_size_2(), convoluted.get_size_3()));
		_bias_gd = move(vec<FLT>(convoluted.get_size_1()));
	}

	_core_dt = FLT(0);
	_bias_dt = FLT(0);
}

cnn_trainy::~cnn_trainy() {}

void cnn_trainy::update(const data<FLT>& _data_prev, FLT alpha, FLT speed)
{
	const tns<FLT>& input = (const tns<FLT>&)_data_prev;

	convolute_rev(input, _core_gd, _core_dt_temp, _convo);
	_core_dt_temp *= speed;
	_core_dt *= alpha;
	_core_dt += _core_dt_temp;

	convolute_rev_colla(input, _bias_gd, _bias_dt_temp, _convo);
	_bias_dt_temp *= speed;
	_bias_dt *= alpha;
	_bias_dt += _bias_dt_temp;
}

void cnn_trainy::update(const nn_trainy& _data_prev, FLT alpha, FLT speed)
{
	update(((const cnn_trainy&)_data_prev)._activ, alpha, speed);
}