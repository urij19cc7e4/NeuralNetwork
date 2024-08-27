#pragma once

#include <random>
#include "nn_params.h"

class rand_init_base
{
protected:
	std::mt19937_64 _rand_gen;

public:
	rand_init_base() : _rand_gen((std::random_device())()) {}

	rand_init_base(const rand_init_base& o) = delete;

	rand_init_base(rand_init_base&& o) = delete;

	virtual ~rand_init_base() {}

	virtual FLT operator()() = 0;
};

template <nn_params::nn_init_t init>
class rand_init;

template <>
class rand_init<nn_params::nn_init_t::normal> : public rand_init_base
{
protected:
	std::normal_distribution<FLT> _distributor;

public:
	rand_init() = delete;

	rand_init(uint64_t isize, uint64_t osize)
		: rand_init_base(), _distributor(
			((FLT)0),
			sqrt(((FLT)2) / (FLT)(isize + osize))) {}

	rand_init(const rand_init& o) = delete;

	rand_init(rand_init&& o) = delete;

	virtual ~rand_init() {}

	virtual FLT operator()() override
	{
		return _distributor(this->_rand_gen);
	}
};

template <>
class rand_init<nn_params::nn_init_t::uniform> : public rand_init_base
{
protected:
	std::uniform_real_distribution<FLT> _distributor;

public:
	rand_init() = delete;

	rand_init(uint64_t isize, uint64_t osize)
		: rand_init_base(), _distributor(
			sqrt(((FLT)6) / (FLT)(isize + osize))* ((FLT)-1),
			sqrt(((FLT)6) / (FLT)(isize + osize))* ((FLT)1)) {}

	rand_init(const rand_init& o) = delete;

	rand_init(rand_init&& o) = delete;

	virtual ~rand_init() {}

	virtual FLT operator()() override
	{
		return _distributor(this->_rand_gen);
	}
};