//#pragma once
//
//#include "cnn.h"
//#include "fnn.h"
//
//#include "info.h"
//#include "pipe.h"
//
//class fnn_train
//{
//private:
//	fnn* _layers;
//	uint64_t _count;
//
//	inline void add_info(std::list<info>* data, pipe<info>* data_pipe, const info& _info);
//	bool check_set(const std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t>& set);
//	void split_set(std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t>& src_set,
//		std::tuple<vec<FLT>*, vec<FLT>*, uint64_t>& train_set,
//		std::tuple<vec<FLT>*, vec<FLT>*, uint64_t>& test_set);
//	void unite_set(std::tuple<vec<FLT>*, vec<FLT>*, uint64_t>& dst_set,
//		std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t>& train_set,
//		std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t>& test_set);
//	bool use_cross_train_mode(std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t>& train_set,
//		std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t>& test_set, bool _split_set,
//		std::tuple<vec<FLT>*, vec<FLT>*, uint64_t>& train_set_cross,
//		std::tuple<vec<FLT>*, vec<FLT>*, uint64_t>& test_set_cross);
//
//public:
//	fnn_train() noexcept;
//	fnn_train(uint64_t isize, uint64_t osize, uint64_t ssize, uint64_t count, nn_params::nn_activ_t activ,
//		nn_params::nn_init_t init = nn_params::nn_init_t::normal, FLT param = (FLT)1);
//	fnn_train(std::initializer_list<uint64_t> sizes, nn_params::nn_activ_t activ,
//		nn_params::nn_init_t init = nn_params::nn_init_t::normal, FLT param = (FLT)1);
//	fnn_train(const nn_params::fnn_layer_info* info, uint64_t count);
//	fnn_train(const fnn_train& o);
//	fnn_train(fnn_train&& o) noexcept;
//	~fnn_train();
//
//	inline uint64_t get_count() const noexcept;
//	uint64_t get_param_count() const noexcept;
//	inline bool is_empty() const noexcept;
//
//	vec<FLT> pass_fwd(const vec<FLT>& input) const;
//	vec<FLT> pass_fwd(vec<FLT>&& input) const;
//
//	void train_batch_mode
//	(
//		std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t> train_set,
//		std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t> test_set = { nullptr, nullptr, (uint64_t)0 },
//		std::list<info>* errors = nullptr,
//		pipe<info>* error_pipe = nullptr,
//		uint64_t max_epochs = (uint64_t)10000,
//		uint64_t max_overs = (uint64_t)75,
//		uint64_t test_freq = (uint64_t)25,
//		bool _split_set = false,
//		FLT convergence = (FLT)1,
//		FLT alpha_start = (FLT)0.90,
//		FLT alpha_end = (FLT)0.00,
//		FLT speed_start = (FLT)0.90,
//		FLT speed_end = (FLT)0.01,
//		FLT max_error = (FLT)0.50,
//		FLT min_error = (FLT)1e-9
//	);
//
//	void train_stoch_mode
//	(
//		std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t> train_set,
//		std::tuple<const vec<FLT>*, const vec<FLT>*, uint64_t> test_set = { nullptr, nullptr, (uint64_t)0 },
//		std::list<info>* errors = nullptr,
//		pipe<info>* error_pipe = nullptr,
//		uint64_t max_epochs = (uint64_t)10000,
//		uint64_t max_overs = (uint64_t)75,
//		uint64_t test_freq = (uint64_t)25,
//		bool _split_set = false,
//		FLT convergence = (FLT)1,
//		FLT alpha_start = (FLT)0.90,
//		FLT alpha_end = (FLT)0.00,
//		FLT speed_start = (FLT)0.90,
//		FLT speed_end = (FLT)0.01,
//		FLT max_error = (FLT)0.50,
//		FLT min_error = (FLT)1e-9
//	);
//
//	fnn_train& operator=(const fnn_train& o);
//	fnn_train& operator=(fnn_train&& o) noexcept;
//};