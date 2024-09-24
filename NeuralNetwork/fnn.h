#pragma once

#include <random>
#include "nn.h"

class fnn;
class fnn_trainy;
class fnn_trainy_batch;
struct fnn_info;

class fnn : public nn
{
private:
	mtx<FLT> _link;
	vec<FLT> _bias;

	nn_params::nn_activ_params _params;

public:
	fnn() noexcept;
	fnn(uint64_t isize, uint64_t osize, const nn_params::nn_activ_params& params, nn_params::nn_init_t init);
	fnn(const mtx<FLT>& link, const vec<FLT>& bias, const nn_params::nn_activ_params& params);
	fnn(mtx<FLT>&& link, vec<FLT>&& bias, const nn_params::nn_activ_params& params) noexcept;
	fnn(std::ifstream& file);
	fnn(const fnn_info& i);
	fnn(const fnn& o);
	fnn(fnn&& o) noexcept;
	virtual ~fnn();

	virtual nn* create_new() const;
	virtual void save_to_file(std::ofstream& file) const;

	inline const nn_params::nn_activ_params& get_activ_params() const noexcept;

	inline uint64_t get_isize() const noexcept;
	inline uint64_t get_osize() const noexcept;

	inline const mtx<FLT>& get_link() const noexcept;
	inline const vec<FLT>& get_bias() const noexcept;

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

	fnn& operator=(const fnn& o);
	fnn& operator=(fnn&& o) noexcept;
};

class fnn_trainy : public nn_trainy
{
protected:
	vec<FLT> _activ;
	vec<FLT> _deriv;

	mtx<FLT> _link_dt;
	vec<FLT> _bias_dt;

	vec<FLT> _link_gd;
	vec<FLT> _bias_gd;

	std::bernoulli_distribution _distributor;
	std::mt19937_64 _rand_gen;
	vec<bool> _drop_map;
	double _drop_out;

	friend class fnn;
	friend class cnn_2_fnn;
	friend class fnn_trainy_batch;

public:
	fnn_trainy() = delete;
	fnn_trainy(uint64_t isize, uint64_t osize, double drop_out, bool delta_hold);
	fnn_trainy(const fnn_trainy& o) = delete;
	fnn_trainy(fnn_trainy&& o) = delete;
	virtual ~fnn_trainy();

	virtual void update(const data<FLT>& _data_prev, FLT alpha, FLT speed);
	virtual void update(const nn_trainy& _data_prev, FLT alpha, FLT speed);
};

class fnn_trainy_batch : public nn_trainy_batch
{
protected:
	mtx<FLT> _link_dt;
	vec<FLT> _bias_dt;

	friend class fnn;

public:
	fnn_trainy_batch() = delete;
	fnn_trainy_batch(uint64_t isize, uint64_t osize);
	fnn_trainy_batch(const fnn_trainy_batch& o) = delete;
	fnn_trainy_batch(fnn_trainy_batch&& o) = delete;
	virtual ~fnn_trainy_batch();

	virtual void begin_update(FLT alpha);

	virtual void update(const nn_trainy& _data, const data<FLT>& _data_prev, FLT speed);
	virtual void update(const nn_trainy& _data, const nn_trainy& _data_prev, FLT speed);
};

struct fnn_info : nn_info
{
public:
	uint64_t isize;
	uint64_t osize;
	nn_params::nn_activ_params params;
	nn_params::nn_init_t init;

	fnn_info() = delete;
	fnn_info(uint64_t isize, uint64_t osize,
		const nn_params::nn_activ_params& params, nn_params::nn_init_t init) noexcept;
	fnn_info(const fnn_info& o) = default;
	fnn_info(fnn_info&& o) = default;
	~fnn_info() = default;

	virtual nn* create_new() const;
};