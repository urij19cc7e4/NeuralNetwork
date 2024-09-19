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

		switch (init)
		{
		case nn_init_t::normal:
			randomizer = (rand_init_base*)(new rand_init<nn_init_t::normal>(isize, osize));
			break;

		case nn_init_t::uniform:
			randomizer = (rand_init_base*)(new rand_init<nn_init_t::uniform>(isize, osize));
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

fnn::fnn(ifstream& file) : fnn()
{
	uint64_t isize;
	uint64_t osize;
	nn_activ_t activ;
	double scale_x;
	double scale_y;
	double scale_z;

	file.read(reinterpret_cast<char*>(&isize), sizeof(isize));
	file.read(reinterpret_cast<char*>(&osize), sizeof(osize));
	file.read(reinterpret_cast<char*>(&activ), sizeof(activ));
	file.read(reinterpret_cast<char*>(&scale_x), sizeof(scale_x));
	file.read(reinterpret_cast<char*>(&scale_y), sizeof(scale_y));
	file.read(reinterpret_cast<char*>(&scale_z), sizeof(scale_z));

	_link = move(mtx<FLT>(osize, isize));
	_bias = move(vec<FLT>(osize));

	_activ = activ;
	_scale_x = (FLT)scale_x;
	_scale_y = (FLT)scale_y;
	_scale_z = (FLT)scale_z;

	file.read(reinterpret_cast<char*>(&_link((uint64_t)0)), sizeof(_link((uint64_t)0)) * _link.get_size());
	file.read(reinterpret_cast<char*>(&_bias((uint64_t)0)), sizeof(_bias((uint64_t)0)) * _bias.get_size());
}

fnn::fnn(const fnn_info& i)
	: fnn(i.isize, i.osize, i.activ, i.init, i.scale_x, i.scale_y, i.scale_z) {}

fnn::fnn(const fnn& o)
	: _link(o._link), _bias(o._bias), _activ(o._activ), _scale_x(o._scale_x), _scale_y(o._scale_y), _scale_z(o._scale_z) {}

fnn::fnn(fnn&& o) noexcept
	: _link(move(o._link)), _bias(move(o._bias)), _activ(o._activ), _scale_x(o._scale_x), _scale_y(o._scale_y), _scale_z(o._scale_z) {}

fnn::~fnn() {}

nn* fnn::create_new() const
{
	return (nn*)(new fnn(*this));
}

void fnn::save_to_file(ofstream& file) const
{
	const uint64_t id = (uint64_t)layer_id::FNN;
	file.write(reinterpret_cast<const char*>(&id), sizeof(id));

	uint64_t isize = get_isize();
	uint64_t osize = get_osize();
	nn_activ_t activ = get_activ_type();
	double scale_x = (double)get_scale_x();
	double scale_y = (double)get_scale_y();
	double scale_z = (double)get_scale_z();

	file.write(reinterpret_cast<const char*>(&isize), sizeof(isize));
	file.write(reinterpret_cast<const char*>(&osize), sizeof(osize));
	file.write(reinterpret_cast<const char*>(&activ), sizeof(activ));
	file.write(reinterpret_cast<const char*>(&scale_x), sizeof(scale_x));
	file.write(reinterpret_cast<const char*>(&scale_y), sizeof(scale_y));
	file.write(reinterpret_cast<const char*>(&scale_z), sizeof(scale_z));

	file.write(reinterpret_cast<const char*>(&_link((uint64_t)0)), sizeof(_link((uint64_t)0)) * _link.get_size());
	file.write(reinterpret_cast<const char*>(&_bias((uint64_t)0)), sizeof(_bias((uint64_t)0)) * _bias.get_size());
}

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

nn_trainy* fnn::get_trainy(const ::data<FLT>& _data_prev, double _drop_out, bool _delta_hold) const
{
	return (nn_trainy*)(new fnn_trainy(_link.get_size_2(), _link.get_size_1(), _drop_out, _delta_hold));
}

nn_trainy* fnn::get_trainy(const nn_trainy& _data_prev, double _drop_out, bool _delta_hold) const
{
	return (nn_trainy*)(new fnn_trainy(_link.get_size_2(), _link.get_size_1(), _drop_out, _delta_hold));
}

nn_trainy_batch* fnn::get_trainy_batch(const ::data<FLT>& _data_prev) const
{
	return (nn_trainy_batch*)(new fnn_trainy_batch(_link.get_size_2(), _link.get_size_1()));
}

nn_trainy_batch* fnn::get_trainy_batch(const nn_trainy& _data_prev) const
{
	return (nn_trainy_batch*)(new fnn_trainy_batch(_link.get_size_2(), _link.get_size_1()));
}

::data<FLT>* fnn::pass_fwd(const ::data<FLT>& _data) const
{
	if (is_empty())
		throw exception(error_msg::fnn_empty_error);
	else
	{
		vec<FLT>* result_ptr = (vec<FLT>*)(new vec<FLT>(move(multiply(_link, (const vec<FLT>&)_data))));

		for (uint64_t i = (uint64_t)0; i < result_ptr->get_size(); ++i)
			(*result_ptr)(i) = activation((*result_ptr)(i) + _bias(i), _scale_x, _scale_y, _scale_z, _activ);

		return (::data<FLT>*)result_ptr;
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

		for (uint64_t i = (uint64_t)0; i < fnn_data._link_gd.get_size(); ++i)
		{
			FLT loss = res_data(i) - fnn_data._activ(i);
			error += loss * loss;

			if (fnn_data._drop_map(i))
			{
				fnn_data._link_gd(i) = fnn_data._deriv(i) * loss;
				fnn_data._bias_gd(i) = loss;
			}
			else
			{
				fnn_data._link_gd(i) = (FLT)0;
				fnn_data._bias_gd(i) = (FLT)0;
			}
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
		multiply_bwd(_link, ((const fnn_trainy&)_data)._link_gd, fnn_data._drop_map, fnn_data._link_gd);

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

		if (fnn_data._drop_out != (double)0.0)
			for (uint64_t i = (uint64_t)0; i < fnn_data._drop_map.get_size(); ++i)
				fnn_data._drop_map(i) = fnn_data._distributor(fnn_data._rand_gen);

		multiply_fwd(_link, (const vec<FLT>&)_data_prev, fnn_data._drop_map, fnn_data._activ);

		for (uint64_t i = (uint64_t)0; i < fnn_data._activ.get_size(); ++i)
			if (fnn_data._drop_map(i))
			{
				FLT value = fnn_data._activ(i) + _bias(i);

				fnn_data._activ(i) = activation(value, _scale_x, _scale_y, _scale_z, _activ);
				fnn_data._deriv(i) = derivation(value, _scale_x, _scale_y, _scale_z, _activ);
			}
			else
			{
				fnn_data._activ(i) = (FLT)0;
				fnn_data._deriv(i) = (FLT)0;
			}

		if (fnn_data._drop_out != (double)0.0)
			fnn_data._activ /= (FLT)((double)1.0 - fnn_data._drop_out);
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

void fnn::train_upd(const nn_trainy_batch& _data)
{
	if (is_empty())
		throw exception(error_msg::fnn_empty_error);
	else
	{
		const fnn_trainy_batch& fnn_data = (const fnn_trainy_batch&)_data;

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

fnn_trainy::fnn_trainy(uint64_t isize, uint64_t osize, double drop_out, bool delta_hold)
	: _activ(osize), _deriv(osize), _link_dt(), _bias_dt(), _link_gd(osize), _bias_gd(osize),
	_rand_gen((random_device())()), _distributor((double)1.0 - drop_out), _drop_map(osize), _drop_out(drop_out)
{
	_drop_map = true;

	if (delta_hold)
	{
		_link_dt = move(mtx<FLT>(osize, isize));
		_bias_dt = move(vec<FLT>(osize));

		_link_dt = (FLT)0;
		_bias_dt = (FLT)0;
	}
}

fnn_trainy::~fnn_trainy() {}

void fnn_trainy::update(const ::data<FLT>& _data_prev, FLT alpha, FLT speed)
{
	const vec<FLT>& input_data = (const vec<FLT>&)_data_prev;

	for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < _link_dt.get_size_1(); ++i)
	{
		for (uint64_t k = (uint64_t)0; k < _link_dt.get_size_2(); ++j, ++k)
			_link_dt(j) = _link_dt(j) * alpha + _link_gd(i) * input_data(k) * speed;

		_bias_dt(i) = _bias_dt(i) * alpha + _bias_gd(i) * speed;
	}
}

void fnn_trainy::update(const nn_trainy& _data_prev, FLT alpha, FLT speed)
{
	update(((const fnn_trainy&)_data_prev)._activ, alpha, speed);
}

fnn_trainy_batch::fnn_trainy_batch(uint64_t isize, uint64_t osize) : _link_dt(osize, isize), _bias_dt(osize)
{
	_link_dt = (FLT)0;
	_bias_dt = (FLT)0;
}

fnn_trainy_batch::~fnn_trainy_batch() {}

void fnn_trainy_batch::begin_update(FLT alpha)
{
	_link_dt *= alpha;
	_bias_dt *= alpha;
}

void fnn_trainy_batch::update(const nn_trainy& _data, const ::data<FLT>& _data_prev, FLT speed)
{
	const fnn_trainy& fnn_data = (const fnn_trainy&)_data;
	const vec<FLT>& input_data = (const vec<FLT>&)_data_prev;

	for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < _link_dt.get_size_1(); ++i)
	{
		for (uint64_t k = (uint64_t)0; k < _link_dt.get_size_2(); ++j, ++k)
			_link_dt(j) += fnn_data._link_gd(i) * input_data(k) * speed;

		_bias_dt(i) += fnn_data._bias_gd(i) * speed;
	}
}

void fnn_trainy_batch::update(const nn_trainy& _data, const nn_trainy& _data_prev, FLT speed)
{
	update(_data, ((const fnn_trainy&)_data_prev)._activ, speed);
}

fnn_info::fnn_info(uint64_t isize, uint64_t osize, nn_activ_t activ, nn_init_t init, FLT scale_x, FLT scale_y, FLT scale_z)
	: isize(isize), osize(osize), activ(activ), init(init), scale_x(scale_x), scale_y(scale_y), scale_z(scale_z) {}

nn* fnn_info::create_new() const
{
	return (nn*)(new fnn(*this));
}