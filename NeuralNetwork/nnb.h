#pragma once

#include <memory>

#include "nn.h"
#include "cnn.h"
#include "fnn.h"

#include "info.h"
#include "pipe.h"

class nnb
{
private:
	std::unique_ptr<std::unique_ptr<nn>[]> _lays;
	uint64_t _size;

public:
	nnb() noexcept;
	nnb(std::initializer_list<std::unique_ptr<nn_params::nn_info>> lays_info);
	nnb(const nnb& o) = delete;
	nnb(nnb&& o) noexcept;
	~nnb();

	uint64_t get_param_count() const noexcept;
	inline uint64_t get_size() const noexcept;
	inline bool is_empty() const noexcept;

	void pass_fwd(data<FLT>& _data) const;

	nnb& operator=(const nnb& o) = delete;
	nnb& operator=(nnb&& o) noexcept;
};