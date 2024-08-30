#include "nnb.h"

#include <cmath>
#include <exception>
#include <random>
#include <utility>
#include <omp.h>

#include "light_appx.h"
#include "rand_sel.h"

using namespace std;
using namespace arithmetic;
using namespace nn_params;

inline void nnb::add_info(list<info>*data_list,pipe<info>*data_pipe,info&&value)
{
	if(data_list!=nullptr)
		data_list->push_back(value);

	if(data_pipe!=nullptr)
		data_pipe->push(value);
}

inline bool nnb::check_mode(const train_mode&gen,const train_mode&tar)
{
	return ((uint64_t)gen&(uint64_t)tar)==(uint64_t)tar;
}

nnb::nnb() noexcept : _lays(nullptr), _size((uint64_t)0) {}

nnb::nnb(initializer_list<unique_ptr<nn_info>> lays_info) : _lays(nullptr), _size(lays_info.size())
{
	if (_size != (uint64_t)0)
	{
		_lays = move(unique_ptr<unique_ptr<nn>[]>(new unique_ptr<nn>[_size]()));
		const unique_ptr<nn_info>* elems = lays_info.begin();

		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			_lays[i] = move(unique_ptr<nn>(elems[i]->create_new()));
	}
}

nnb::nnb(const nnb& o) : _lays(nullptr), _size(o._size)
{
	if (_size != (uint64_t)0)
	{
		_lays = move(unique_ptr<unique_ptr<nn>[]>(new unique_ptr<nn>[_size]()));

		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			_lays[i] = move(unique_ptr<nn>(o._lays[i]->create_new()));
	}
}

nnb::nnb(nnb&& o) noexcept : _lays(move(o._lays)), _size(o._size)
{
	o._size = (uint64_t)0;
}

nnb::~nnb() {}

uint64_t nnb::get_param_count() const noexcept
{
	if (is_empty())
		return (uint64_t)0;
	else
	{
		uint64_t result = (uint64_t)0;

		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			result += _lays[i]->get_param_count();

		return result;
	}
}

inline uint64_t nnb::get_size() const noexcept
{
	return _size;
}

inline bool nnb::is_empty() const noexcept
{
	return _lays == nullptr;
}

::data<FLT>* nnb::pass_fwd(const ::data<FLT>&_data) const
{
	if (is_empty())
		throw exception(error_msg::nnb_empty_error);
	else
	{
		::data<FLT>*result_ptr=_lays[(uint64_t)0]->pass_fwd(_data);

		for (uint64_t i = (uint64_t)1; i < _size; ++i)
		{
			::data<FLT>*temp=_lays[i]->pass_fwd(*result_ptr);
			delete result_ptr;
			result_ptr=temp;
		}

		return result_ptr;
	}
}

void nnb::train_stoch_mode
(
	data_set train_set,
	data_set test_set,
	train_mode mode,
	std::list<info>* errors,
	pipe<info>* error_pipe,
	uint64_t max_epochs,
	uint64_t max_overs,
	uint64_t test_freq,
	FLT convergence,
	FLT alpha_start,
	FLT alpha_end,
	FLT speed_start,
	FLT speed_end,
	FLT max_error,
	FLT min_error
)
{
	if (is_empty())
		throw exception(error_msg::nnb_empty_error);
	else
	{
		add_info(errors,error_pipe,move(info(max_epochs,msg_type::count_epo)));
		add_info(errors,error_pipe,move(info(train_set.size,msg_type::count_set_1)));
		add_info(errors,error_pipe,move(info(test_set.size,msg_type::count_set_2)));
		add_info(errors,error_pipe,move(info(msg_type::stoch_mode)));

		if(check_mode(mode,train_mode::CROSS_TEST))
			add_info(errors,error_pipe,move(info(msg_type::cross_mode)));

		bool max_epo_reached=true;

		uint64_t err_overs=(uint64_t)0;
		uint64_t last_layer=_size-(uint64_t)1;

		FLT min_test_error=(FLT)0;
		FLT prev_test_error=(FLT)0;

		light_appx alpha_appx(alpha_start,alpha_end,max_epochs);
		light_appx speed_appx(speed_start,speed_end,max_epochs);
		light_appx conv_appx(max(convergence*(FLT)_size,(FLT)1),(FLT)1,max_epochs);

		unique_ptr<rand_sel_i<uint64_t>>selector=train_set.size>(uint64_t)250
			?unique_ptr<rand_sel_i<uint64_t>>((new rand_sel<uint64_t,false>(train_set.size - (uint64_t)1, (uint64_t)0)))
			:unique_ptr<rand_sel_i<uint64_t>>((new rand_sel<uint64_t,true>(train_set.size - (uint64_t)1, (uint64_t)0)));

		unique_ptr<unique_ptr<nn_trainy>[]>train_data(new unique_ptr<nn_trainy>[_size]);

		train_data[(uint64_t)0]=move(unique_ptr<nn_trainy>(_lays[(uint64_t)0]->get_trainy((*train_set.idata[(uint64_t)0]))));
		for(uint64_t i=(uint64_t)1;i<_size;++i)
			train_data[i]=move(unique_ptr<nn_trainy>(_lays[i]->get_trainy(*(train_data[i-(uint64_t)1]))));

		for(uint64_t i=(uint64_t)0;i<max_epochs;++i)
		{
			FLT train_err=(FLT)0,test_err=(FLT)0;
			FLT alpha_epo=alpha_appx.forward(),speed_epo=speed_appx.forward();

			light_appx speed_lay_appxer(conv_appx.forward(),speed_epo,_size,(FLT)0,(FLT)2);
			selector->reset();

			for(uint64_t j=(uint64_t)0;j<train_set.size;++j)
			{
				uint64_t num=selector->next();

				_lays[(uint64_t)0]->train_fwd(*(train_data[(uint64_t)0]),(*train_set.idata[num]));
				for(uint64_t k=(uint64_t)1;k<_size;++k)
					_lays[k]->train_fwd(*(train_data[k]),*(train_data[k-(uint64_t)1]));

				train_err+=_lays[last_layer]->train_bwd(*(train_data[last_layer]),(*train_set.odata[num]));
				for(uint64_t k=last_layer;k>(uint64_t)0;--k)
					_lays[k]->train_bwd(*(train_data[k]),*(train_data[k-(uint64_t)1]));

				speed_lay_appxer.reset();

				{
					FLT speed_lay=speed_lay_appxer.forward();

					train_data[(uint64_t)0]->update((*train_set.idata[num]),alpha_epo,speed_lay);
					_lays[(uint64_t)0]->train_upd(*(train_data[(uint64_t)0]));
				}

				for(uint64_t k=(uint64_t)1;k<_size;++k)
				{
					FLT speed_lay=speed_lay_appxer.forward();

					train_data[k]->update(*(train_data[k-(uint64_t)1]),alpha_epo,speed_lay);
					_lays[k]->train_upd(*(train_data[k]));
				}
			}

			train_err/=(FLT)train_set.size;

			if(train_err<min_error)
			{
				add_info(errors,error_pipe,move(info(i+(uint64_t)1,msg_type::min_err_reached)));
				max_epo_reached=false;

				break;
			}

			if(check_mode(mode,train_mode::CROSS_TEST))
			{
			if(i%test_freq==(uint64_t)0)
			{
				for(uint64_t j=(uint64_t)0;j<test_set.size;++j)
				{
					unique_ptr<::data<FLT>> fwd_result(pass_fwd((*test_set.idata[j])));

					for(uint64_t k=(uint64_t)0;k<fwd_result->get_size();++k)
					{
						FLT loss=(*test_set.odata[j])[k] -(*fwd_result)(k);
						test_err+=loss*loss;
					}
				}

				test_err/=(FLT)test_set.size;
				prev_test_error=test_err;
			}
			else
				test_err=prev_test_error;

			if(i==(uint64_t)0||min_test_error>test_err)
				min_test_error=test_err;

			if(test_err>min_test_error*(max_error+(FLT)1))
				++err_overs;

			if(test_err>min_test_error*(max_error*(FLT)2+(FLT)1)||err_overs>max_overs)
			{
				add_info(errors,error_pipe,move(info(i+(uint64_t)1,msg_type::max_err_reached)));
				max_epo_reached=false;

				break;
			}
			}

			add_info(errors,error_pipe,move(info(train_err,test_err)));
		}

		if(max_epo_reached)
			add_info(errors,error_pipe,move(info(msg_type::max_epo_reached)));
	}
}

nnb& nnb::operator=(const nnb& o)
{
	if (o.is_empty())
	{
		_lays = nullptr;
		_size = (uint64_t)0;
	}
	else
	{
		_lays = move(unique_ptr<unique_ptr<nn>[]>(new unique_ptr<nn>[o._size]()));
		_size = o._size;

		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			_lays[i] = move(unique_ptr<nn>(o._lays[i]->create_new()));
	}

	return *this;
}

nnb& nnb::operator=(nnb&& o) noexcept
{
	_lays = move(o._lays);
	_size = o._size;

	o._size = (uint64_t)0;

	return *this;
}