#include "fnn.h"

#include "rand_init.h"

using namespace std;
using namespace arithmetic;
using namespace nn_params;

fnn::fnn() noexcept
	: _link(), _bias(), _activ(nn_activ_t::__count__), _scale_x((FLT)1), _scale_y((FLT)1), _scale_z((FLT)1) {}

fnn::fnn(uint64_t isize, uint64_t osize, nn_activ_t activ, nn_init_t init, FLT scale_x, FLT scale_y, FLT scale_z)
	: _link(osize, isize), _bias(osize), _activ(activ), _scale_x(scale_x), _scale_y(scale_y), _scale_z(scale_z)
{
	if (isize == (uint64_t)0 || osize == (uint64_t)0 || scale_x == (FLT)0 || scale_y == (FLT)0 || scale_z == (FLT)0
		|| activ >= nn_activ_t::__count__ || init >= nn_init_t::__count__)
		throw exception(error_msg::fnn_wrong_init_error);
	else
	{
		rand_init_base* randomizer = nullptr;

		__assume(init < nn_init_t::__count__);
		switch (init)
		{
		case nn_init_t::normal:
			randomizer = new rand_init<nn_init_t::normal>(isize, osize);
			break;

		case nn_init_t::uniform:
			randomizer = new rand_init<nn_init_t::uniform>(isize, osize);
			break;

		default:
			__assume(false);
		}

		for (uint64_t i = (uint64_t)0; i < _link.get_size(); ++i)
			_link(i) = (*randomizer)();

		_bias = (FLT)0;

		delete randomizer;
	}
}

fnn::fnn(const mtx<FLT>& link, const vec<FLT>& bias, nn_activ_t activ, FLT scale_x, FLT scale_y, FLT scale_z)
	: _link(link), _bias(bias), _activ(activ), _scale_x(scale_x), _scale_y(scale_y), _scale_z(scale_z) {}

fnn::fnn(mtx<FLT>&& link, vec<FLT>&& bias, nn_activ_t activ, FLT scale_x, FLT scale_y, FLT scale_z) noexcept
	: _link(move(link)), _bias(move(bias)), _activ(activ), _scale_x(scale_x), _scale_y(scale_y), _scale_z(scale_z) {}

fnn::fnn(const fnn& o)
	: _link(o._link), _bias(o._bias), _activ(o._activ), _scale_x(o._scale_x), _scale_y(o._scale_y), _scale_z(o._scale_z) {}

fnn::fnn(fnn&& o) noexcept
	: _link(move(o._link)), _bias(move(o._bias)), _activ(o._activ), _scale_x(o._scale_x), _scale_y(o._scale_y), _scale_z(o._scale_z) {}

fnn::~fnn() {}

inline nn_activ_t fnn::get_activ_type() const noexcept
{
	return _activ;
}

inline FLT fnn::get_scale_x() const noexcept
{
	return _scale_x;
}

inline FLT fnn::get_scale_y() const noexcept
{
	return _scale_y;
}

inline FLT fnn::get_scale_z() const noexcept
{
	return _scale_z;
}

inline uint64_t fnn::get_isize() const noexcept
{
	return _link.get_size_2();
}

inline uint64_t fnn::get_osize() const noexcept
{
	return _link.get_size_1();
}

inline const mtx<FLT>& fnn::get_link() const noexcept
{
	return _link;
}

inline const vec<FLT>& fnn::get_bias() const noexcept
{
	return _bias;
}

inline bool fnn::is_empty() const noexcept
{
	return _link.is_empty() || _bias.is_empty();
}

uint64_t fnn::get_param_count() const noexcept
{
	return _link.get_size() + _bias.get_size();
}

nn_trainy* fnn::get_trainy(const ::data<FLT>& _data_prev) const
{
	return new fnn_trainy(_link.get_size_2(), _link.get_size_1());
}

nn_trainy* fnn::get_trainy(const nn_trainy& _data_prev) const
{
	return new fnn_trainy(_link.get_size_2(), _link.get_size_1());
}

void fnn::pass_fwd(::data<FLT>& _data) const
{
	if (is_empty())
		throw exception(error_msg::fnn_empty_error);
	else
	{
		vec<FLT>& result = (vec<FLT>&)_data;
		result = move(multiply(_link, result));

		__assume(result.get_size() == _bias.get_size());
		for (uint64_t i = (uint64_t)0; i < result.get_size(); ++i)
			result(i) = activation(result(i) + _bias(i), _scale_x, _scale_y, _scale_z, _activ);
	}
}

FLT fnn::train_bwd(nn_trainy& _data, const ::data<FLT>& _data_next) const
{
	if (is_empty())
		throw exception(error_msg::fnn_empty_error);
	else
	{
		FLT error(0);
		fnn_trainy& fnn_data = (fnn_trainy&)_data;
		const vec<FLT>& res_data = (const vec<FLT>&)_data_next;

		__assume(fnn_data._link_gd.get_size() == res_data.get_size());
		__assume(fnn_data._link_gd.get_size() == fnn_data._activ.get_size());
		__assume(fnn_data._link_gd.get_size() == fnn_data._deriv.get_size());
		__assume(fnn_data._link_gd.get_size() == fnn_data._bias_gd.get_size());
		for (uint64_t i = (uint64_t)0; i < fnn_data._link_gd.get_size(); ++i)
		{
			FLT loss = res_data(i) - fnn_data._activ(i);

			fnn_data._link_gd(i) = fnn_data._deriv(i) * loss;
			fnn_data._bias_gd(i) = loss;
			error += loss * loss;
		}

		return error;
	}
}

void fnn::train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const
{
	if (is_empty())
		throw exception(error_msg::fnn_empty_error);
	else
	{
		fnn_trainy& fnn_data = (fnn_trainy&)_data_prev;
		multiply_bwd(_link, ((const fnn_trainy&)_data)._link_gd, fnn_data._link_gd);

		fnn_data._bias_gd = fnn_data._link_gd;
		fnn_data._link_gd *= fnn_data._deriv;
	}
}

void fnn::train_fwd(nn_trainy& _data, const ::data<FLT>& _data_prev) const
{
	if (is_empty())
		throw exception(error_msg::fnn_empty_error);
	else
	{
		fnn_trainy& fnn_data = (fnn_trainy&)_data;
		multiply_fwd(_link, (const vec<FLT>&)_data_prev, fnn_data._activ);

		__assume(fnn_data._activ.get_size() == _bias.get_size());
		__assume(fnn_data._deriv.get_size() == _bias.get_size());
		for (uint64_t i = (uint64_t)0; i < fnn_data._activ.get_size(); ++i)
		{
			FLT value = fnn_data._activ(i) + _bias(i);

			fnn_data._activ(i) = activation(value, _scale_x, _scale_y, _scale_z, _activ);
			fnn_data._deriv(i) = derivation(value, _scale_x, _scale_y, _scale_z, _activ);
		}
	}
}

void fnn::train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const
{
	train_fwd(_data, ((const fnn_trainy&)_data_prev)._activ);
}

void fnn::train_upd(const nn_trainy& _data)
{
	if (is_empty())
		throw exception(error_msg::fnn_empty_error);
	else
	{
		const fnn_trainy& fnn_data = (const fnn_trainy&)_data;

		_link += fnn_data._link_dt;
		_bias += fnn_data._bias_dt;
	}
}

fnn& fnn::operator=(const fnn& o)
{
	_link = o._link;
	_bias = o._bias;
	_activ = o._activ;
	_scale_x = o._scale_x;
	_scale_y = o._scale_y;
	_scale_z = o._scale_z;

	return *this;
}

fnn& fnn::operator=(fnn&& o) noexcept
{
	_link = move(o._link);
	_bias = move(o._bias);
	_activ = o._activ;
	_scale_x = o._scale_x;
	_scale_y = o._scale_y;
	_scale_z = o._scale_z;

	return *this;
}

fnn_trainy::fnn_trainy(uint64_t isize, uint64_t osize)
	: _activ(osize), _deriv(osize), _link_dt(osize, isize), _bias_dt(osize), _link_gd(osize), _bias_gd(osize)
{
	_link_dt = FLT(0);
	_bias_dt = FLT(0);
}

fnn_trainy::~fnn_trainy() {}

void fnn_trainy::update(const ::data<FLT>& _data_prev, FLT alpha, FLT speed)
{
	const vec<FLT>& input = (const vec<FLT>&)_data_prev;

	__assume(_link_dt.get_size_1() == _bias_dt.get_size());
	__assume(_link_dt.get_size_1() == _link_gd.get_size());
	__assume(_link_dt.get_size_1() == _bias_gd.get_size());
	for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < _link_dt.get_size_1(); ++i)
	{
		__assume(_link_dt.get_size_2() == input.get_size());
		for (uint64_t k = (uint64_t)0; k < _link_dt.get_size_2(); ++j, ++k)
			_link_dt(j) = _link_dt(j) * alpha + _link_gd(i) * input(k) * speed;

		_bias_dt(i) = _bias_dt(i) * alpha + _bias_gd(i) * speed;
	}
}

void fnn_trainy::update(const nn_trainy& _data_prev, FLT alpha, FLT speed)
{
	update(((const fnn_trainy&)_data_prev)._activ, alpha, speed);
}