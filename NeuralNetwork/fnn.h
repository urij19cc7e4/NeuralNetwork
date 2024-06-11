#pragma once

#include <cstdint>
#include <tuple>

#include "nn_params.h"
#include "info.h"
#include "pipe.h"
#include "rfm.h"

class fnnl
{
private:
	mtx<double> _weights;
	vec<double> _bias_ws;

	nn_params::nn_activ_t _activ;
	double _param;

public:
	fnnl() noexcept;
	fnnl(uint64_t isize, uint64_t osize, nn_params::nn_activ_t activ, nn_params::nn_init_t init, double param);
	fnnl(const mtx<double>& weights, const vec<double>& bias_ws, nn_params::nn_activ_t activ, double param);
	fnnl(mtx<double>&& weights, vec<double>&& bias_ws, nn_params::nn_activ_t activ, double param) noexcept;
	fnnl(const fnnl& o);
	fnnl(fnnl&& o) noexcept;
	~fnnl();

	vec<double> pass_bwd(const vec<double>& output) const;
	vec<double> pass_fwd(const vec<double>& input) const;
	std::tuple<vec<double>, vec<double>> train_bwd(const vec<double>& grads, const vec<double>& grads_bias) const;
	std::tuple<vec<double>, vec<double>> train_fwd(const vec<double>& input) const;
	void train_upd(const mtx<double>& weights_ds, const vec<double>& bias_ws_ds);

	inline nn_params::nn_activ_t get_activ_type() const noexcept;
	inline uint64_t get_isize() const noexcept;
	inline uint64_t get_osize() const noexcept;
	inline double get_param() const noexcept;
	inline uint64_t get_param_count() const noexcept;
	inline bool is_empty() const noexcept;

	fnnl& operator=(const fnnl&o);
	fnnl& operator=(fnnl&&o) noexcept;
};

class fnn
{
private:
	fnnl* _layers;
	uint64_t _count;

	inline void add_info(std::list<info>* data, pipe<info>* data_pipe, const info& _info);
	bool check_set(const std::tuple<const vec<double>*, const vec<double>*, uint64_t>& set);
	void split_set(std::tuple<const vec<double>*, const vec<double>*, uint64_t>& src_set,
		std::tuple<vec<double>*, vec<double>*, uint64_t>& train_set,
		std::tuple<vec<double>*, vec<double>*, uint64_t>& test_set);
	void unite_set(std::tuple<vec<double>*, vec<double>*, uint64_t>& dst_set,
		std::tuple<const vec<double>*, const vec<double>*, uint64_t>& train_set,
		std::tuple<const vec<double>*, const vec<double>*, uint64_t>& test_set);
	bool use_cross_train_mode(std::tuple<const vec<double>*, const vec<double>*, uint64_t>& train_set,
		std::tuple<const vec<double>*, const vec<double>*, uint64_t>& test_set, bool _split_set,
		std::tuple<vec<double>*, vec<double>*, uint64_t>& train_set_cross,
		std::tuple<vec<double>*, vec<double>*, uint64_t>& test_set_cross);

public:
	fnn() noexcept;
	fnn(uint64_t isize, uint64_t osize, uint64_t ssize, uint64_t count, nn_params::nn_activ_t activ,
		nn_params::nn_init_t init = nn_params::nn_init_t::normal, double param = (double)1);
	fnn(std::initializer_list<uint64_t> sizes, nn_params::nn_activ_t activ,
		nn_params::nn_init_t init = nn_params::nn_init_t::normal, double param = (double)1);
	fnn(const nn_params::fnn_layer_info* info, uint64_t count);
	fnn(const fnn& o);
	fnn(fnn&& o) noexcept;
	~fnn();

	vec<double> pass_bwd(const vec<double>& output) const;
	vec<double> pass_bwd(vec<double>&& output) const;
	vec<double> pass_fwd(const vec<double>& input) const;
	vec<double> pass_fwd(vec<double>&& input) const;

	void train_batch_mode
	(
		std::tuple<const vec<double>*, const vec<double>*, uint64_t> train_set,
		std::tuple<const vec<double>*, const vec<double>*, uint64_t> test_set = { nullptr, nullptr, (uint64_t)0 },
		std::list<info>* errors = nullptr,
		pipe<info>* error_pipe = nullptr,
		uint64_t max_epochs = (uint64_t)10000,
		uint64_t max_overs = (uint64_t)75,
		uint64_t test_freq = (uint64_t)25,
		bool _split_set = false,
		double convergence = (double)1,
		double alpha_start = (double)0.90,
		double alpha_end = (double)0.00,
		double speed_start = (double)0.90,
		double speed_end = (double)0.01,
		double max_error = (double)0.10,
		double min_error = (double)1e-9
	);

	void train_stoch_mode
	(
		std::tuple<const vec<double>*, const vec<double>*, uint64_t> train_set,
		std::tuple<const vec<double>*, const vec<double>*, uint64_t> test_set = { nullptr, nullptr, (uint64_t)0 },
		std::list<info>* errors = nullptr,
		pipe<info>* error_pipe = nullptr,
		uint64_t max_epochs = (uint64_t)10000,
		uint64_t max_overs = (uint64_t)75,
		uint64_t test_freq = (uint64_t)25,
		bool _split_set = false,
		double convergence = (double)1,
		double alpha_start = (double)0.90,
		double alpha_end = (double)0.00,
		double speed_start = (double)0.90,
		double speed_end = (double)0.01,
		double max_error = (double)0.10,
		double min_error = (double)1e-9
	);

	inline uint64_t get_count() const noexcept;
	uint64_t get_param_count() const noexcept;
	inline bool is_empty() const noexcept;

	fnn& operator=(const fnn&o);
	fnn& operator=(fnn&&o) noexcept;
};