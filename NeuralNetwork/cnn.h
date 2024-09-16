#pragma once

#include "nn.h"

class cnn;
class cnn_trainy;
class cnn_trainy_batch;

class cnn : public nn
{
private:
	tns<FLT> _core;
	vec<FLT> _bias;

	FLT _bn_link;
	FLT _bn_bias;

	nn_params::nn_activ_t _activ;
	FLT _scale_x;
	FLT _scale_y;
	FLT _scale_z;

	bool _pool;

public:
	cnn() noexcept;
	cnn(uint64_t count, uint64_t depth, uint64_t height, uint64_t width, nn_params::nn_activ_t activ,
		nn_params::nn_init_t init, FLT scale_x, FLT scale_y, FLT scale_z, bool pool);
	cnn(const tns<FLT>& core, const vec<FLT>& bias, FLT bn_link, FLT bn_bias, nn_params::nn_activ_t activ,
		FLT scale_x, FLT scale_y, FLT scale_z, bool pool);
	cnn(tns<FLT>&& core, vec<FLT>&& bias, FLT bn_link, FLT bn_bias, nn_params::nn_activ_t activ,
		FLT scale_x, FLT scale_y, FLT scale_z, bool pool) noexcept;
	cnn(const nn_params::cnn_info&i);
	cnn(const cnn& o);
	cnn(cnn&& o) noexcept;
	virtual ~cnn();

	virtual nn*create_new() const;

	inline nn_params::nn_activ_t get_activ_type() const noexcept;
	inline FLT get_scale_x() const noexcept;
	inline FLT get_scale_y() const noexcept;
	inline FLT get_scale_z() const noexcept;
	inline FLT get_bn_link() const noexcept;
	inline FLT get_bn_bias() const noexcept;
	inline bool get_pool_enabled() const noexcept;
	inline uint64_t get_count() const noexcept;
	inline uint64_t get_depth() const noexcept;
	inline uint64_t get_height() const noexcept;
	inline uint64_t get_width() const noexcept;
	inline uint64_t get_size() const noexcept;
	inline const tns<FLT>& get_core() const noexcept;
	inline const vec<FLT>& get_bias() const noexcept;
	inline bool is_empty() const noexcept;

	virtual uint64_t get_param_count() const noexcept;
	virtual nn_trainy* get_trainy(const data<FLT>& _data_prev, bool _drop_out, bool _delta_hold) const;
	virtual nn_trainy* get_trainy(const nn_trainy& _data_prev, bool _drop_out, bool _delta_hold) const;
	virtual nn_trainy_batch* get_trainy_batch(const data<FLT>& _data_prev) const;
	virtual nn_trainy_batch* get_trainy_batch(const nn_trainy& _data_prev) const;

	virtual data<FLT>*pass_fwd(const data<FLT>&_data) const;
	virtual FLT train_bwd(nn_trainy& _data, const data<FLT>& _data_next) const;
	virtual void train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const;
	virtual void train_fwd(nn_trainy& _data, const data<FLT>& _data_prev) const;
	virtual void train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const;
	virtual void train_upd(const nn_trainy& _data);
	virtual void train_upd(const nn_trainy_batch& _data);

	cnn& operator=(const cnn& o);
	cnn& operator=(cnn&& o) noexcept;
};

class cnn_trainy : public nn_trainy
{
protected:
	tns<FLT> _activ;
	tns<FLT> _deriv;

	tns<FLT> _core_dt;
	tns<FLT> _core_dt_temp;
	vec<FLT> _bias_dt;
	vec<FLT> _bias_dt_temp;

	FLT _bn_link_dt;
	FLT _bn_bias_dt;

	tns<FLT> _core_gd;
	vec<FLT> _bias_gd;

	FLT _bn_link_gd;
	FLT _bn_bias_gd;

	tns<uint8_t> _pool_map;
	tns<FLT> _pool_temp_1;
	tns<FLT> _pool_temp_2;

	bool _pool;
	bool _drop_out;

	friend class cnn;
	friend class cnn_2_fnn;
	friend class cnn_trainy_batch;

public:
	cnn_trainy() = delete;
	cnn_trainy(uint64_t count, uint64_t depth, uint64_t h_data, uint64_t w_data,
		uint64_t h_core, uint64_t w_core, bool pool, bool drop_out, bool delta_hold);
	cnn_trainy(const cnn_trainy& o) = delete;
	cnn_trainy(cnn_trainy&& o) = delete;
	virtual ~cnn_trainy();

	virtual void update(const data<FLT>& _data_prev, FLT alpha, FLT speed);
	virtual void update(const nn_trainy& _data_prev, FLT alpha, FLT speed);
};

class cnn_trainy_batch : public nn_trainy_batch
{
protected:
	tns<FLT> _core_dt;
	tns<FLT> _core_dt_temp;
	vec<FLT> _bias_dt;
	vec<FLT> _bias_dt_temp;

	FLT _bn_link_dt;
	FLT _bn_bias_dt;

	friend class cnn;

public:
	cnn_trainy_batch() = delete;
	cnn_trainy_batch(uint64_t count, uint64_t depth, uint64_t h_core, uint64_t w_core);
	cnn_trainy_batch(const cnn_trainy_batch& o) = delete;
	cnn_trainy_batch(cnn_trainy_batch&& o) = delete;
	virtual ~cnn_trainy_batch();

	virtual void begin_update(FLT alpha);

	virtual void update(const nn_trainy& _data, const data<FLT>& _data_prev, FLT speed);
	virtual void update(const nn_trainy& _data, const nn_trainy& _data_prev, FLT speed);
};