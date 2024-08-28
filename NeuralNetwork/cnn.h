#pragma once

#include "nn.h"

class cnn;
class cnn_trainy;

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

	nn_params::nn_convo_t _convo;
	bool _pool;

public:
	cnn() noexcept;
	cnn(uint64_t height, uint64_t width, uint64_t size, nn_params::nn_activ_t activ, nn_params::nn_init_t init,
		FLT scale_x, FLT scale_y, FLT scale_z, nn_params::nn_convo_t convo, bool pool);
	cnn(const tns<FLT>& core, const vec<FLT>& bias, FLT bn_link, FLT bn_bias, nn_params::nn_activ_t activ,
		FLT scale_x, FLT scale_y, FLT scale_z, nn_params::nn_convo_t convo, bool pool);
	cnn(tns<FLT>&& core, vec<FLT>&& bias, FLT bn_link, FLT bn_bias, nn_params::nn_activ_t activ,
		FLT scale_x, FLT scale_y, FLT scale_z, nn_params::nn_convo_t convo, bool pool) noexcept;
	cnn(const cnn& o);
	cnn(cnn&& o) noexcept;
	virtual ~cnn();

	inline nn_params::nn_activ_t get_activ_type() const noexcept;
	inline nn_params::nn_convo_t get_convo_type() const noexcept;
	inline FLT get_scale_x() const noexcept;
	inline FLT get_scale_y() const noexcept;
	inline FLT get_scale_z() const noexcept;
	inline FLT get_bn_link() const noexcept;
	inline FLT get_bn_bias() const noexcept;
	inline bool get_pool_enabled() const noexcept;
	inline uint64_t get_height() const noexcept;
	inline uint64_t get_width() const noexcept;
	inline uint64_t get_size() const noexcept;
	inline const tns<FLT>& get_core() const noexcept;
	inline const vec<FLT>& get_bias() const noexcept;
	inline bool is_empty() const noexcept;

	virtual uint64_t get_param_count() const noexcept;
	virtual nn_trainy* get_trainy(const data<FLT>& _data_prev) const;
	virtual nn_trainy* get_trainy(const nn_trainy& _data_prev) const;

	virtual void pass_fwd(data<FLT>& _data) const;
	virtual FLT train_bwd(nn_trainy& _data, const data<FLT>& _data_next) const;
	virtual void train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const;
	virtual void train_fwd(nn_trainy& _data, const data<FLT>& _data_prev) const;
	virtual void train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const;
	virtual void train_upd(const nn_trainy& _data);

	cnn& operator=(const cnn& o);
	cnn& operator=(cnn&& o) noexcept;
};

class cnn_trainy : public nn_trainy
{
private:
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

	nn_params::nn_convo_t _convo;
	bool _pool;

	friend class cnn;
	friend class cnn_2_fnn;

public:
	cnn_trainy() = delete;
	cnn_trainy(const tns<FLT>& data, const tns<FLT>& core, nn_params::nn_convo_t convo, bool pool);
	cnn_trainy(const cnn_trainy& o) = delete;
	cnn_trainy(cnn_trainy&& o) = delete;
	virtual ~cnn_trainy();

	virtual void update(const data<FLT>& _data_prev, FLT alpha, FLT speed);
	virtual void update(const nn_trainy& _data_prev, FLT alpha, FLT speed);
};