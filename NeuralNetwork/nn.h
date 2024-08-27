#pragma once

#include "nn_rfm.h"

class nn;
class nn_trainy;

class nn
{
public:
	virtual uint64_t get_param_count() const noexcept = 0;
	virtual nn_trainy* get_trainy(const data<FLT>& _data_prev) const = 0;
	virtual nn_trainy* get_trainy(const nn_trainy& _data_prev) const = 0;

	virtual void pass_fwd(data<FLT>& _data) const = 0;
	virtual FLT train_bwd(nn_trainy& _data, const data<FLT>& _data_next) const = 0;
	virtual void train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const = 0;
	virtual void train_fwd(nn_trainy& _data, const data<FLT>& _data_prev) const = 0;
	virtual void train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const = 0;
	virtual void train_upd(const nn_trainy& _data) = 0;
};

class nn_trainy
{
public:
	virtual void update(const data<FLT>& _data_prev, FLT alpha, FLT speed) = 0;
	virtual void update(const nn_trainy& _data_prev, FLT alpha, FLT speed) = 0;
};