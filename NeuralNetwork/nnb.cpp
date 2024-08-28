#include "nnb.h"

#include <cmath>
#include <exception>
#include <random>
#include <utility>
#include <omp.h>

#include "light_appx.h"
#include "rand_sel.h"

using namespace std;
using namespace arithmetic;
using namespace nn_params;

nnb::nnb() noexcept : _lays(nullptr), _size((uint64_t)0) {}

nnb::nnb(initializer_list<unique_ptr<nn_info>> lays_info)
	: _lays(new unique_ptr<nn>[lays_info.size()]()), _size(lays_info.size())
{

}

nnb::nnb(nnb&& o) noexcept : _lays(move(o._lays)), _size(o._size)
{
	o._size = (uint64_t)0;
}

nnb::~nnb() {}

uint64_t nnb::get_param_count() const noexcept
{
	if (is_empty())
		return (uint64_t)0;
	else
	{
		uint64_t result = (uint64_t)0;

		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			result += _lays[i]->get_param_count();

		return result;
	}
}

inline uint64_t nnb::get_size() const noexcept
{
	return _size;
}

inline bool nnb::is_empty() const noexcept
{
	return _lays == nullptr;
}

void nnb::pass_fwd(data<FLT>& _data) const
{
	for (uint64_t i = (uint64_t)0; i < _size; ++i)
		_lays[i]->pass_fwd(_data);
}

nnb& nnb::operator=(nnb&& o) noexcept
{
	_lays = move(o._lays);
	_size = o._size;

	o._size = (uint64_t)0;

	return *this;
}