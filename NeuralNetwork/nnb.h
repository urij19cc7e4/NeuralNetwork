#pragma once

#include <memory>
#include <string>

#include "nn.h"
#include "cnn.h"
#include "fnn.h"
#include "cnn_2_fnn.h"

#include "info.h"
#include "pipe.h"

enum class train_mode : uint64_t
{
	NONE = 0x0000000000000000,

	BATCH = 0x0000000000000001,
	MINI_BATCH = 0x0000000000000002,
	STOCHASTIC = 0x0000000000000004,

	RANDOM_TRAIN_SELECTION = 0x0000000000000010,
	STRICT_TRAIN_SELECTION = 0x0000000000000020,
	RANDOM_TEST_SELECTION = 0x0000000000000040,
	STRICT_TEST_SELECTION = 0x0000000000000080,

	CROSS_TEST = 0x0000000000000100,
	DROP_OUT = 0x0000000000000200
};

inline train_mode operator&(const train_mode& lhs, const train_mode& rhs);
inline train_mode operator|(const train_mode& lhs, const train_mode& rhs);
inline train_mode operator~(const train_mode& arg);

struct data_set
{
public:
	std::vector<std::shared_ptr<data<FLT>>> idata;
	std::vector<std::shared_ptr<data<FLT>>> odata;
	uint64_t size;
};

extern data_set empty_set;

struct phase
{
public:
	train_mode mode;
	uint64_t train_batch_size;
	uint64_t test_batch_size;
	uint64_t epoch_count;
	FLT drop_out_rate;
	FLT speed_scale;
	FLT alpha_start;
	FLT alpha_end;
	FLT speed_start;
	FLT speed_end;
	FLT max_error;
	FLT min_error;

	phase() = delete;
	phase
	(
		train_mode mode = train_mode::NONE,
		uint64_t train_batch_size = (uint64_t)10,
		uint64_t test_batch_size = (uint64_t)100,
		uint64_t epoch_count = (uint64_t)900,
		FLT drop_out_rate = (FLT)0.01,
		FLT speed_scale = (FLT)1.00,
		FLT alpha_start = (FLT)0.90,
		FLT alpha_end = (FLT)0.10,
		FLT speed_start = (FLT)0.90,
		FLT speed_end = (FLT)0.10,
		FLT max_error = (FLT)0.50,
		FLT min_error = (FLT)1e-5
	);
	phase(const phase& o) = default;
	phase(phase&& o) = default;
	~phase() = default;
};

class nnb
{
private:
	nn** _lays;
	uint64_t _size;

	static inline void add_info(std::list<info>* value_list, pipe<info>* value_pipe, info&& value);
	static inline bool check_mode(const train_mode& gen, const train_mode& tar);
	static std::list<phase> get_valid_phases(const std::list<phase>& phs, uint64_t train_size, uint64_t test_size);
	static std::string get_header(uint64_t layers, uint64_t params);

public:
	nnb() noexcept;
	nnb(std::initializer_list<std::unique_ptr<nn_info>> lays_info);
	nnb(std::ifstream& file);
	nnb(const nnb& o);
	nnb(nnb&& o) noexcept;
	~nnb();

	void save_to_file(std::ofstream& file) const;

	uint64_t get_param_count() const noexcept;
	inline uint64_t get_size() const noexcept;
	inline bool is_empty() const noexcept;

	data<FLT>* pass(const data<FLT>& _data) const;

	void train
	(
		data_set train_set = empty_set,
		data_set test_set = empty_set,
		train_mode mode = train_mode::NONE,
		const std::list<phase>* phase_list = nullptr,
		std::list<info>* error_list = nullptr,
		pipe<info>* error_pipe = nullptr,
		uint64_t train_batch_size = (uint64_t)10,
		uint64_t test_batch_size = (uint64_t)100,
		uint64_t epoch_count = (uint64_t)900,
		uint64_t fail_count = (uint64_t)75,
		uint64_t test_freq = (uint64_t)25,
		FLT drop_out_rate = (FLT)0.01,
		FLT speed_scale = (FLT)1.00,
		FLT alpha_start = (FLT)0.90,
		FLT alpha_end = (FLT)0.10,
		FLT speed_start = (FLT)0.90,
		FLT speed_end = (FLT)0.10,
		FLT max_error = (FLT)0.50,
		FLT min_error = (FLT)1e-5
	);

	nnb& operator=(const nnb& o);
	nnb& operator=(nnb&& o) noexcept;
};