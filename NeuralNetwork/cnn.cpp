#include "cnn.h"

#include "rand_init.h"

using namespace std;
using namespace arithmetic;
using namespace nn_params;

cnn::cnn() noexcept
	: _core(), _bias(), _bn_link((FLT)1), _bn_bias((FLT)0), _params(nn_activ_t::__count__), _pool(false)
{}

cnn::cnn(uint64_t count, uint64_t depth, uint64_t height, uint64_t width, const nn_activ_params& params, nn_init_t init, bool pool)
	: _core(count * depth, height, width), _bias(count), _bn_link((FLT)1), _bn_bias((FLT)0), _params(params), _pool(pool)
{
	if (count == (uint64_t)0 || depth == (uint64_t)0 || height == (uint64_t)0 || width == (uint64_t)0 || init >= nn_init_t::__count__
		|| params.activ_type >= nn_activ_t::__count__ || params.scale_x == (FLT)0 || params.scale_y == (FLT)0)
		throw exception(error_msg::cnn_wrong_init_error);
	else
	{
		rand_init_base* randomizer = nullptr;

		switch (init)
		{
		case nn_init_t::normal:
			randomizer = (rand_init_base*)(new rand_init<nn_init_t::normal>(depth * height * width, (uint64_t)0));
			break;

		case nn_init_t::uniform:
			randomizer = (rand_init_base*)(new rand_init<nn_init_t::uniform>(depth * height * width, (uint64_t)0));
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

cnn::cnn(const tns<FLT>& core, const vec<FLT>& bias, FLT bn_link, FLT bn_bias, const nn_activ_params& params, bool pool)
	: _core(core), _bias(bias), _bn_link(bn_link), _bn_bias(bn_bias), _params(params), _pool(pool)
{}

cnn::cnn(tns<FLT>&& core, vec<FLT>&& bias, FLT bn_link, FLT bn_bias, const nn_activ_params& params, bool pool) noexcept
	: _core(move(core)), _bias(move(bias)), _bn_link(bn_link), _bn_bias(bn_bias), _params(params), _pool(pool)
{}

cnn::cnn(ifstream& file) : cnn()
{
	uint64_t count;
	uint64_t depth;
	uint64_t height;
	uint64_t width;
	uint64_t pool;
	nn_activ_params params(nn_activ_t::__count__);

	file.read(reinterpret_cast<char*>(&count), sizeof(count));
	file.read(reinterpret_cast<char*>(&depth), sizeof(depth));
	file.read(reinterpret_cast<char*>(&height), sizeof(height));
	file.read(reinterpret_cast<char*>(&width), sizeof(width));
	file.read(reinterpret_cast<char*>(&_params), sizeof(_params));
	file.read(reinterpret_cast<char*>(&pool), sizeof(pool));

	_core = move(tns<FLT>(count * depth, height, width));
	_bias = move(vec<FLT>(count));
	_pool = (bool)pool;

	_params.activ_type = params.activ_type;
	_params.scale_x = params.scale_x;
	_params.scale_y = params.scale_y;
	_params.shift_x = params.shift_x;
	_params.shift_y = params.shift_y;

	file.read(reinterpret_cast<char*>(&_core((uint64_t)0)), sizeof(_core((uint64_t)0)) * _core.get_size());
	file.read(reinterpret_cast<char*>(&_bias((uint64_t)0)), sizeof(_bias((uint64_t)0)) * _bias.get_size());

	double bn_link;
	double bn_bias;

	file.read(reinterpret_cast<char*>(&bn_link), sizeof(bn_link));
	file.read(reinterpret_cast<char*>(&bn_bias), sizeof(bn_bias));

	_bn_link = (FLT)bn_link;
	_bn_bias = (FLT)bn_bias;
}

cnn::cnn(const cnn_info& i)
	: cnn(i.count, i.depth, i.height, i.width, i.params, i.init, i.pool)
{}

cnn::cnn(const cnn& o)
	: _core(o._core), _bias(o._bias), _bn_link(o._bn_link), _bn_bias(o._bn_bias), _params(o._params), _pool(o._pool)
{}

cnn::cnn(cnn&& o) noexcept
	: _core(move(o._core)), _bias(move(o._bias)), _bn_link(o._bn_link), _bn_bias(o._bn_bias), _params(o._params), _pool(o._pool)
{}

cnn::~cnn() {}

nn* cnn::create_new() const
{
	return (nn*)(new cnn(*this));
}

void cnn::save_to_file(ofstream& file) const
{
	const uint64_t id = (uint64_t)layer_id::CNN;
	file.write(reinterpret_cast<const char*>(&id), sizeof(id));

	uint64_t count(get_count());
	uint64_t depth(get_depth());
	uint64_t height(get_height());
	uint64_t width(get_width());
	nn_activ_params params(get_activ_params());
	uint64_t pool((uint64_t)get_pool_enabled());

	file.write(reinterpret_cast<const char*>(&count), sizeof(count));
	file.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
	file.write(reinterpret_cast<const char*>(&height), sizeof(height));
	file.write(reinterpret_cast<const char*>(&width), sizeof(width));
	file.write(reinterpret_cast<const char*>(&params), sizeof(params));
	file.write(reinterpret_cast<const char*>(&pool), sizeof(pool));

	file.write(reinterpret_cast<const char*>(&_core((uint64_t)0)), sizeof(_core((uint64_t)0)) * _core.get_size());
	file.write(reinterpret_cast<const char*>(&_bias((uint64_t)0)), sizeof(_bias((uint64_t)0)) * _bias.get_size());

	double bn_link((double)get_bn_link());
	double bn_bias((double)get_bn_bias());

	file.write(reinterpret_cast<const char*>(&bn_link), sizeof(bn_link));
	file.write(reinterpret_cast<const char*>(&bn_bias), sizeof(bn_bias));
}

inline const nn_params::nn_activ_params& cnn::get_activ_params() const noexcept
{
	return _params;
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

inline uint64_t cnn::get_count() const noexcept
{
	return _bias.get_size();
}

inline uint64_t cnn::get_depth() const noexcept
{
	return _core.get_size_1() / _bias.get_size();
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

nn_trainy* cnn::get_trainy(const ::data<FLT>& _data_prev, double _drop_out, bool _delta_hold) const
{
	const tns<FLT>& cnn_data = (const tns<FLT>&)_data_prev;
	return (nn_trainy*)(new cnn_trainy(_bias.get_size(), cnn_data.get_size_1(), _core.get_size_2(),
		_core.get_size_3(), cnn_data.get_size_2(), cnn_data.get_size_3(), _pool, _drop_out, _delta_hold));
}

nn_trainy* cnn::get_trainy(const nn_trainy& _data_prev, double _drop_out, bool _delta_hold) const
{
	const tns<FLT>& cnn_data = (const tns<FLT>&)(((const cnn_trainy&)_data_prev)._activ);
	return (nn_trainy*)(new cnn_trainy(_bias.get_size(), cnn_data.get_size_1(), _core.get_size_2(),
		_core.get_size_3(), cnn_data.get_size_2(), cnn_data.get_size_3(), _pool, _drop_out, _delta_hold));
}

nn_trainy_batch* cnn::get_trainy_batch(const ::data<FLT>& _data_prev) const
{
	const tns<FLT>& cnn_data = (const tns<FLT>&)_data_prev;
	return (nn_trainy_batch*)(new cnn_trainy_batch(_bias.get_size(),
		cnn_data.get_size_1(), _core.get_size_2(), _core.get_size_3()));
}

nn_trainy_batch* cnn::get_trainy_batch(const nn_trainy& _data_prev) const
{
	const tns<FLT>& cnn_data = (const tns<FLT>&)(((const cnn_trainy&)_data_prev)._activ);
	return (nn_trainy_batch*)(new cnn_trainy_batch(_bias.get_size(),
		cnn_data.get_size_1(), _core.get_size_2(), _core.get_size_3()));
}

::data<FLT>* cnn::pass_fwd(const ::data<FLT>& _data) const
{
	if (is_empty())
		throw exception(error_msg::cnn_empty_error);
	else
	{
		tns<FLT>* result_ptr = (tns<FLT>*)(new tns<FLT>(move(convolute((const tns<FLT>&)_data, _core))));
		uint64_t layer_size = result_ptr->get_size_2() * result_ptr->get_size_3();

		for (uint64_t i = (uint64_t)0; i < result_ptr->get_size(); ++i)
			(*result_ptr)(i) = activation((*result_ptr)(i) + _bias(i / layer_size), _params);

		if (_pool)
			*result_ptr = move(pool(*result_ptr));

		return (::data<FLT>*)result_ptr;
	}
}

FLT cnn::train_bwd(nn_trainy& _data, const ::data<FLT>& _data_next) const
{
	if (is_empty())
		throw exception(error_msg::cnn_empty_error);
	else
	{
		FLT error(0);
		cnn_trainy& cnn_data = (cnn_trainy&)_data;
		const tns<FLT>& res_data = (const tns<FLT>&)_data_next;
		uint64_t layer_size = res_data.get_size_2() * res_data.get_size_3();

		if (cnn_data._pool)
		{
			for (uint64_t i = (uint64_t)0; i < cnn_data._pool_temp_2.get_size(); ++i)
			{
				FLT loss = res_data(i) - cnn_data._activ(i);
				error += loss * loss;

				if (cnn_data._drop_map(i / layer_size))
					cnn_data._pool_temp_2(i) = loss;
				else
					cnn_data._pool_temp_2(i) = (FLT)0;
			}

			pool_bwd(cnn_data._pool_temp_2, cnn_data._pool_map, cnn_data._core_gd);
			collapse(cnn_data._core_gd, cnn_data._bias_gd);
			cnn_data._core_gd *= cnn_data._deriv;
		}
		else
		{
			cnn_data._bias_gd = (FLT)0;

			for (uint64_t i = (uint64_t)0; i < cnn_data._core_gd.get_size(); ++i)
			{
				FLT loss = res_data(i) - cnn_data._activ(i);
				error += loss * loss;

				if (cnn_data._drop_map(i / layer_size))
				{
					cnn_data._core_gd(i) = cnn_data._deriv(i) * loss;
					cnn_data._bias_gd(i / layer_size) += loss;
				}
				else
					cnn_data._core_gd(i) = (FLT)0;
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

		if (cnn_data._pool)
		{
			convolute_bwd(((const cnn_trainy&)_data)._core_gd, _core, cnn_data._drop_map, cnn_data._pool_temp_2);
			pool_bwd(cnn_data._pool_temp_2, cnn_data._pool_map, cnn_data._core_gd);
		}
		else
			convolute_bwd(((const cnn_trainy&)_data)._core_gd, _core, cnn_data._drop_map, cnn_data._core_gd);

		collapse(cnn_data._core_gd, cnn_data._bias_gd);
		cnn_data._core_gd *= cnn_data._deriv;
	}
}

void cnn::train_fwd(nn_trainy& _data, const ::data<FLT>& _data_prev) const
{
	if (is_empty())
		throw exception(error_msg::cnn_empty_error);
	else
	{
		cnn_trainy& cnn_data = (cnn_trainy&)_data;

		if (cnn_data._drop_out != (double)0.0)
			for (uint64_t i = (uint64_t)0; i < cnn_data._drop_map.get_size(); ++i)
				cnn_data._drop_map(i) = cnn_data._distributor(cnn_data._rand_gen);

		if (cnn_data._pool)
		{
			convolute_fwd((const tns<FLT>&)_data_prev, _core, cnn_data._drop_map, cnn_data._pool_temp_1);
			uint64_t layer_size = cnn_data._pool_temp_1.get_size_2() * cnn_data._pool_temp_1.get_size_3();

			for (uint64_t i = (uint64_t)0; i < cnn_data._pool_temp_1.get_size(); ++i)
				if (cnn_data._drop_map(i / layer_size))
				{
					FLT value = cnn_data._pool_temp_1(i) + _bias(i / layer_size);

					cnn_data._pool_temp_1(i) = activation(value, _params);
					cnn_data._deriv(i) = derivation(value, _params);
				}
				else
				{
					cnn_data._pool_temp_1(i) = (FLT)0;
					cnn_data._deriv(i) = (FLT)0;
				}

			pool_fwd(cnn_data._pool_temp_1, cnn_data._pool_map, cnn_data._activ);
		}
		else
		{
			convolute_fwd((const tns<FLT>&)_data_prev, _core, cnn_data._drop_map, cnn_data._activ);
			uint64_t layer_size = cnn_data._activ.get_size_2() * cnn_data._activ.get_size_3();

			for (uint64_t i = (uint64_t)0; i < cnn_data._activ.get_size(); ++i)
				if (cnn_data._drop_map(i / layer_size))
				{
					FLT value = cnn_data._activ(i) + _bias(i / layer_size);

					cnn_data._activ(i) = activation(value, _params);
					cnn_data._deriv(i) = derivation(value, _params);
				}
				else
				{
					cnn_data._activ(i) = (FLT)0;
					cnn_data._deriv(i) = (FLT)0;
				}
		}

		if (cnn_data._drop_out != (double)0.0)
			cnn_data._activ /= (FLT)((double)1.0 - cnn_data._drop_out);
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

void cnn::train_upd(const nn_trainy_batch& _data)
{
	if (is_empty())
		throw exception(error_msg::cnn_empty_error);
	else
	{
		const cnn_trainy_batch& cnn_data = (const cnn_trainy_batch&)_data;

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
	_pool = o._pool;

	_params.activ_type = o._params.activ_type;
	_params.scale_x = o._params.scale_x;
	_params.scale_y = o._params.scale_y;
	_params.shift_x = o._params.shift_x;
	_params.shift_y = o._params.shift_y;

	return *this;
}

cnn& cnn::operator=(cnn&& o) noexcept
{
	_core = move(o._core);
	_bias = move(o._bias);
	_bn_link = o._bn_link;
	_bn_bias = o._bn_bias;
	_pool = o._pool;

	_params.activ_type = o._params.activ_type;
	_params.scale_x = o._params.scale_x;
	_params.scale_y = o._params.scale_y;
	_params.shift_x = o._params.shift_x;
	_params.shift_y = o._params.shift_y;

	return *this;
}

cnn_trainy::cnn_trainy(uint64_t count, uint64_t depth, uint64_t h_core, uint64_t w_core,
	uint64_t h_data, uint64_t w_data, bool pool, double drop_out, bool delta_hold)
	: _activ(), _deriv(), _core_dt(), _core_dt_temp(), _bias_dt(), _bias_dt_temp(),
	_core_gd(), _bias_gd(), _rand_gen((random_device())()), _distributor((double)1.0 - drop_out),
	_drop_map(count), _pool_map(), _pool_temp_1(), _pool_temp_2(),
	_bn_link_dt((FLT)0), _bn_bias_dt((FLT)0), _bn_link_gd((FLT)0), _bn_bias_gd((FLT)0),
	_pool(pool), _drop_out(drop_out)
{
	tns<FLT> data_temp(depth, h_data, w_data);
	tns<FLT> core_temp(count * depth, h_core, w_core);
	tns<FLT> conv_temp(move(convolute(data_temp, core_temp)));
	_drop_map = true;

	if (pool)
	{
		tns<FLT> pool_temp(move(arithmetic::pool(conv_temp)));

		_activ = move(tns<FLT>(pool_temp.get_size_1(), pool_temp.get_size_2(), pool_temp.get_size_3()));
		_deriv = move(tns<FLT>(conv_temp.get_size_1(), conv_temp.get_size_2(), conv_temp.get_size_3()));
		_core_gd = move(tns<FLT>(conv_temp.get_size_1(), conv_temp.get_size_2(), conv_temp.get_size_3()));
		_bias_gd = move(vec<FLT>(conv_temp.get_size_1()));

		_pool_map = move(tns<uint8_t>(pool_temp.get_size_1(), pool_temp.get_size_2(), pool_temp.get_size_3()));
		_pool_temp_1 = move(tns<FLT>(conv_temp.get_size_1(), conv_temp.get_size_2(), conv_temp.get_size_3()));
		_pool_temp_2 = move(tns<FLT>(pool_temp.get_size_1(), pool_temp.get_size_2(), pool_temp.get_size_3()));
	}
	else
	{
		_activ = move(tns<FLT>(conv_temp.get_size_1(), conv_temp.get_size_2(), conv_temp.get_size_3()));
		_deriv = move(tns<FLT>(conv_temp.get_size_1(), conv_temp.get_size_2(), conv_temp.get_size_3()));
		_core_gd = move(tns<FLT>(conv_temp.get_size_1(), conv_temp.get_size_2(), conv_temp.get_size_3()));
		_bias_gd = move(vec<FLT>(conv_temp.get_size_1()));
	}

	if (delta_hold)
	{
		_core_dt = move(tns<FLT>(count * depth, h_core, w_core));
		_core_dt_temp = move(tns<FLT>(count * depth, h_core, w_core));
		_bias_dt = move(vec<FLT>(count));
		_bias_dt_temp = move(vec<FLT>(count));

		_core_dt = (FLT)0;
		_bias_dt = (FLT)0;
	}
}

cnn_trainy::~cnn_trainy() {}

void cnn_trainy::update(const ::data<FLT>& _data_prev, FLT alpha, FLT speed)
{
	const tns<FLT>& input_data = (const tns<FLT>&)_data_prev;

	convolute_rev(input_data, _core_gd, _drop_map, _core_dt_temp);
	_core_dt_temp *= speed;
	_core_dt *= alpha;
	_core_dt += _core_dt_temp;

	convolute_rev(input_data, _bias_gd, _drop_map, _bias_dt_temp);
	_bias_dt_temp *= speed;
	_bias_dt *= alpha;
	_bias_dt += _bias_dt_temp;
}

void cnn_trainy::update(const nn_trainy& _data_prev, FLT alpha, FLT speed)
{
	update(((const cnn_trainy&)_data_prev)._activ, alpha, speed);
}

cnn_trainy_batch::cnn_trainy_batch(uint64_t count, uint64_t depth, uint64_t h_core, uint64_t w_core)
	: _core_dt(count * depth, h_core, w_core), _core_dt_temp(count * depth, h_core, w_core),
	_bias_dt(count), _bias_dt_temp(count), _bn_link_dt((FLT)0), _bn_bias_dt((FLT)0)
{
	_core_dt = (FLT)0;
	_bias_dt = (FLT)0;
}

cnn_trainy_batch::~cnn_trainy_batch() {}

void cnn_trainy_batch::begin_update(FLT alpha)
{
	_core_dt *= alpha;
	_bias_dt *= alpha;
}

void cnn_trainy_batch::update(const nn_trainy& _data, const ::data<FLT>& _data_prev, FLT speed)
{
	const cnn_trainy& cnn_data = (const cnn_trainy&)_data;
	const tns<FLT>& input_data = (const tns<FLT>&)_data_prev;

	convolute_rev(input_data, cnn_data._core_gd, cnn_data._drop_map, _core_dt_temp);
	_core_dt_temp *= speed;
	_core_dt += _core_dt_temp;

	convolute_rev(input_data, cnn_data._bias_gd, cnn_data._drop_map, _bias_dt_temp);
	_bias_dt_temp *= speed;
	_bias_dt += _bias_dt_temp;
}

void cnn_trainy_batch::update(const nn_trainy& _data, const nn_trainy& _data_prev, FLT speed)
{
	update(_data, ((const cnn_trainy&)_data_prev)._activ, speed);
}

cnn_info::cnn_info(uint64_t count, uint64_t depth, uint64_t height, uint64_t width,
	const nn_activ_params& params, nn_init_t init, bool pool) noexcept
	: count(count), depth(depth), height(height), width(width), params(params), init(init), pool(pool)
{}

nn* cnn_info::create_new() const
{
	return (nn*)(new cnn(*this));
}