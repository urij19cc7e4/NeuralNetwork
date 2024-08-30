#pragma once

#include "nn.h"

class fnn;
class fnn_trainy;

class fnn : public nn
{
private:
	mtx<FLT> _link;
	vec<FLT> _bias;

	nn_params::nn_activ_t _activ;
	FLT _scale_x;
	FLT _scale_y;
	FLT _scale_z;

public:
	fnn() noexcept;
	fnn(uint64_t isize, uint64_t osize, nn_params::nn_activ_t activ, nn_params::nn_init_t init,
		FLT scale_x, FLT scale_y, FLT scale_z);
	fnn(const mtx<FLT>& link, const vec<FLT>& bias, nn_params::nn_activ_t activ, FLT scale_x, FLT scale_y, FLT scale_z);
	fnn(mtx<FLT>&& link, vec<FLT>&& bias, nn_params::nn_activ_t activ, FLT scale_x, FLT scale_y, FLT scale_z) noexcept;
	fnn(const nn_params::fnn_info&i);
	fnn(const fnn& o);
	fnn(fnn&& o) noexcept;
	virtual ~fnn();

	virtual nn*create_new() const;

	inline nn_params::nn_activ_t get_activ_type() const noexcept;
	inline FLT get_scale_x() const noexcept;
	inline FLT get_scale_y() const noexcept;
	inline FLT get_scale_z() const noexcept;
	inline uint64_t get_isize() const noexcept;
	inline uint64_t get_osize() const noexcept;
	inline const mtx<FLT>& get_link() const noexcept;
	inline const vec<FLT>& get_bias() const noexcept;
	inline bool is_empty() const noexcept;

	virtual uint64_t get_param_count() const noexcept;
	virtual nn_trainy* get_trainy(const data<FLT>& _data_prev) const;
	virtual nn_trainy* get_trainy(const nn_trainy& _data_prev) const;

	virtual data<FLT>*pass_fwd(const data<FLT>&_data) const;
	virtual FLT train_bwd(nn_trainy& _data, const data<FLT>& _data_next) const;
	virtual void train_bwd(const nn_trainy& _data, nn_trainy& _data_prev) const;
	virtual void train_fwd(nn_trainy& _data, const data<FLT>& _data_prev) const;
	virtual void train_fwd(nn_trainy& _data, const nn_trainy& _data_prev) const;
	virtual void train_upd(const nn_trainy& _data);

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

	friend class fnn;
	friend class cnn_2_fnn;

public:
	fnn_trainy() = delete;
	fnn_trainy(uint64_t isize, uint64_t osize);
	fnn_trainy(const fnn_trainy& o) = delete;
	fnn_trainy(fnn_trainy&& o) = delete;
	virtual ~fnn_trainy();

	virtual void update(const data<FLT>& _data_prev, FLT alpha, FLT speed);
	virtual void update(const nn_trainy& _data_prev, FLT alpha, FLT speed);
};