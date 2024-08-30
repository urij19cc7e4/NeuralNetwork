#pragma once

#include "nn.h"
#include "cnn.h"
#include "fnn.h"

class cnn_2_fnn;
class cnn_2_fnn_trainy;

class cnn_2_fnn : public nn
{
private:
	bool _max_pool;

public:
	cnn_2_fnn(bool max_pool = false) noexcept;
	cnn_2_fnn(const nn_params::cnn_2_fnn_info&i) noexcept;
	cnn_2_fnn(const cnn_2_fnn& o) noexcept;
	cnn_2_fnn(cnn_2_fnn&& o) noexcept;
	virtual ~cnn_2_fnn();

	virtual nn*create_new() const;

	virtual uint64_t get_param_count() const noexcept;
	virtual nn_trainy* get_trainy(const data<FLT>& _data_prev) const;
	virtual nn_trainy* get_trainy(const nn_trainy& _data_prev) const;

	virtual data<FLT>*pass_fwd(const data<FLT>&_data) const;
	virtual FLT train_bwd(nn_trainy& _data, const data<FLT>& _data_next) const;
	virtual void train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const;
	virtual void train_fwd(nn_trainy& _data, const data<FLT>& _data_prev) const;
	virtual void train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const;
	virtual void train_upd(const nn_trainy& _data);

	cnn_2_fnn& operator=(const cnn_2_fnn& o) noexcept;
	cnn_2_fnn& operator=(cnn_2_fnn&& o) noexcept;
};

class cnn_2_fnn_trainy : public fnn_trainy
{
private:
	vec<uint64_t> _pool_map;
	bool _max_pool;

	friend class cnn_2_fnn;

public:
	cnn_2_fnn_trainy() = delete;
	cnn_2_fnn_trainy(const tns<FLT>& data, bool max_pool);
	cnn_2_fnn_trainy(const cnn_trainy& o) = delete;
	cnn_2_fnn_trainy(cnn_trainy&& o) = delete;
	virtual ~cnn_2_fnn_trainy();

	virtual void update(const data<FLT>& _data_prev, FLT alpha, FLT speed);
	virtual void update(const nn_trainy& _data_prev, FLT alpha, FLT speed);
};