#pragma once

#include <memory>

#include "nn.h"
#include "cnn.h"
#include "fnn.h"
#include "layer_adapters.h"

#include "info.h"
#include "pipe.h"

enum class train_mode : uint64_t
{
	NONE = 0x0000,
	CROSS_TEST = 0x0001,
	DROP_OUT = 0x0002,
	MT = 0x0004
};

struct data_set
{
	std::unique_ptr<std::unique_ptr<data<FLT>>[]> idata;
	std::unique_ptr<std::unique_ptr<data<FLT>>[]> odata;
	uint64_t size;
};

class nnb
{
private:
	std::unique_ptr<std::unique_ptr<nn>[]> _lays;
	uint64_t _size;

	static inline void add_info(std::list<info>*data_list,pipe<info>*data_pipe,info&&value);
	static inline bool check_mode(const train_mode&gen,const train_mode&tar);

public:
	nnb() noexcept;
	nnb(std::initializer_list<std::unique_ptr<nn_info>> lays_info);
	nnb(const nnb& o);
	nnb(nnb&& o) noexcept;
	~nnb();

	uint64_t get_param_count() const noexcept;
	inline uint64_t get_size() const noexcept;
	inline bool is_empty() const noexcept;

	data<FLT>* pass_fwd(const data<FLT>&_data) const;

	void train
	(
		data_set train_set,
		data_set test_set,
		train_mode mode=train_mode::NONE,
		std::list<info>*errors=nullptr,
		pipe<info>*error_pipe=nullptr,
		uint64_t batch_size=(uint64_t)0,
		uint64_t max_epochs=(uint64_t)5000,
		uint64_t max_overs=(uint64_t)25,
		uint64_t test_freq=(uint64_t)25,
		FLT convergence=(FLT)1.00,
		FLT alpha_start=(FLT)0.90,
		FLT alpha_end=(FLT)0.10,
		FLT speed_start=(FLT)0.90,
		FLT speed_end=(FLT)0.10,
		FLT max_error=(FLT)0.50,
		FLT min_error=(FLT)1e-9
	);

	nnb& operator=(const nnb& o);
	nnb& operator=(nnb&& o) noexcept;
};