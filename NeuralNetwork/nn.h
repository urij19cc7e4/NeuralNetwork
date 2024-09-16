#pragma once

#include "nn_params.h"
#include "nn_rfm.h"

class nn;
class nn_trainy;
class nn_trainy_batch;
struct nn_info;

class nn
{
public:
	virtual nn* create_new() const=0;
	virtual uint64_t get_param_count() const noexcept = 0;

	virtual nn_trainy* get_trainy(const data<FLT>& _data_prev, bool _drop_out, bool _delta_hold) const = 0;
	virtual nn_trainy* get_trainy(const nn_trainy& _data_prev, bool _drop_out, bool _delta_hold) const = 0;
	virtual nn_trainy_batch* get_trainy_batch(const data<FLT>& _data_prev) const = 0;
	virtual nn_trainy_batch* get_trainy_batch(const nn_trainy& _data_prev) const = 0;

	virtual data<FLT>*pass_fwd(const data<FLT>&_data) const=0;
	virtual FLT train_bwd(nn_trainy& _data, const data<FLT>& _data_next) const = 0;
	virtual void train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const = 0;
	virtual void train_fwd(nn_trainy& _data, const data<FLT>& _data_prev) const = 0;
	virtual void train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const = 0;
	virtual void train_upd(const nn_trainy& _data) = 0;
	virtual void train_upd(const nn_trainy_batch& _data) = 0;
};

class nn_trainy
{
public:
	virtual void update(const data<FLT>& _data_prev, FLT alpha, FLT speed) = 0;
	virtual void update(const nn_trainy& _data_prev, FLT alpha, FLT speed) = 0;
};

class nn_trainy_batch
{
public:
	virtual void begin_update(FLT alpha) = 0;

	virtual void update(const nn_trainy& _data, const data<FLT>& _data_prev, FLT speed) = 0;
	virtual void update(const nn_trainy& _data, const nn_trainy& _data_prev, FLT speed) = 0;
};

struct nn_info
{
public:
	virtual nn*create_new() const = 0;
};