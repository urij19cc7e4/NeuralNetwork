#pragma once

#include "nn.h"
#include "cnn.h"
#include "fnn.h"

class cnn_2_fnn;
class cnn_2_fnn_trainy;
class cnn_2_fnn_trainy_batch;
struct cnn_2_fnn_info;

class cnn_2_fnn : public nn
{
private:
	bool _flatten;
	bool _max_pool;

public:
	cnn_2_fnn(bool flatten = true, bool max_pool = true) noexcept;
	cnn_2_fnn(std::ifstream& file) noexcept;
	cnn_2_fnn(const cnn_2_fnn_info& i) noexcept;
	cnn_2_fnn(const cnn_2_fnn& o) noexcept;
	cnn_2_fnn(cnn_2_fnn&& o) noexcept;
	virtual ~cnn_2_fnn();

	virtual nn* create_new() const;
	virtual void save_to_file(std::ofstream& file) const;

	inline bool get_flatten_enabled() const noexcept;
	inline bool get_max_pool_enabled() const noexcept;
	inline bool is_empty() const noexcept;

	virtual uint64_t get_param_count() const noexcept;

	virtual nn_trainy* get_trainy(const data<FLT>& _data_prev, double _drop_out, bool _delta_hold) const;
	virtual nn_trainy* get_trainy(const nn_trainy& _data_prev, double _drop_out, bool _delta_hold) const;

	virtual nn_trainy_batch* get_trainy_batch(const data<FLT>& _data_prev) const;
	virtual nn_trainy_batch* get_trainy_batch(const nn_trainy& _data_prev) const;

	virtual data<FLT>* pass_fwd(const data<FLT>& _data) const;

	virtual FLT train_bwd(nn_trainy& _data, const data<FLT>& _data_next) const;
	virtual void train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const;

	virtual void train_fwd(nn_trainy& _data, const data<FLT>& _data_prev) const;
	virtual void train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const;

	virtual void train_upd(const nn_trainy& _data);
	virtual void train_upd(const nn_trainy_batch& _data);

	cnn_2_fnn& operator=(const cnn_2_fnn& o) noexcept;
	cnn_2_fnn& operator=(cnn_2_fnn&& o) noexcept;
};

class cnn_2_fnn_trainy : public fnn_trainy
{
protected:
	vec<uint64_t> _pool_map;

	bool _flatten;
	bool _max_pool;

	friend class cnn_2_fnn;
	friend class cnn_2_fnn_trainy_batch;

public:
	cnn_2_fnn_trainy() = delete;
	cnn_2_fnn_trainy(const tns<FLT>& data, bool max_pool, bool flatten, double drop_out);
	cnn_2_fnn_trainy(const cnn_trainy& o) = delete;
	cnn_2_fnn_trainy(cnn_trainy&& o) = delete;
	virtual ~cnn_2_fnn_trainy();

	virtual void update(const data<FLT>& _data_prev, FLT alpha, FLT speed);
	virtual void update(const nn_trainy& _data_prev, FLT alpha, FLT speed);
};

class cnn_2_fnn_trainy_batch : public fnn_trainy_batch
{
protected:
	friend class cnn_2_fnn;

public:
	cnn_2_fnn_trainy_batch() = delete;
	cnn_2_fnn_trainy_batch(const tns<FLT>& data);
	cnn_2_fnn_trainy_batch(const cnn_2_fnn_trainy_batch& o) = delete;
	cnn_2_fnn_trainy_batch(cnn_2_fnn_trainy_batch&& o) = delete;
	virtual ~cnn_2_fnn_trainy_batch();

	virtual void begin_update(FLT alpha);

	virtual void update(const nn_trainy& _data, const data<FLT>& _data_prev, FLT speed);
	virtual void update(const nn_trainy& _data, const nn_trainy& _data_prev, FLT speed);
};

struct cnn_2_fnn_info : nn_info
{
public:
	bool flatten;
	bool max_pool;

	cnn_2_fnn_info() = delete;
	cnn_2_fnn_info(bool flatten, bool max_pool) noexcept;
	cnn_2_fnn_info(const cnn_2_fnn_info& o) = default;
	cnn_2_fnn_info(cnn_2_fnn_info&& o) = default;
	~cnn_2_fnn_info() = default;

	virtual nn* create_new() const;
};