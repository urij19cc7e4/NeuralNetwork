#include "info.h"

info::info() noexcept : _data_1((double)0), _data_2((double)0), _data_3((uint64_t)0), _type(msg_type::__count__) {}

info::info(double data_1, double data_2, uint64_t data_3, msg_type type) noexcept : _data_1(data_1), _data_2(data_2), _data_3(data_3), _type(type) {}

info::info(double data_1, double data_2, uint64_t data_3) noexcept : _data_1(data_1), _data_2(data_2), _data_3(data_3), _type(msg_type::data) {}

info::info(double data_1, double data_2, msg_type type) noexcept : _data_1(data_1), _data_2(data_2), _data_3((uint64_t)0), _type(type) {}

info::info(double data_1, double data_2) noexcept : _data_1(data_1), _data_2(data_2), _data_3((uint64_t)0), _type(msg_type::data) {}

info::info(uint64_t data_3, msg_type type) noexcept : _data_1((double)0), _data_2((double)0), _data_3(data_3), _type(type) {}

info::info(uint64_t data_3) noexcept : _data_1((double)0), _data_2((double)0), _data_3(data_3), _type(msg_type::data) {}

info::info(msg_type type) noexcept : _data_1((double)0), _data_2((double)0), _data_3((uint64_t)0), _type(type) {}

info::info(const info& o) noexcept : _data_1(o._data_1), _data_2(o._data_2), _data_3(o._data_3), _type(o._type) {}

info::info(info&& o) noexcept : _data_1(o._data_1), _data_2(o._data_2), _data_3(o._data_3), _type(o._type) {}

info::~info() noexcept {}

double info::get_data_1() const noexcept
{
	return _data_1;
}

double info::get_data_2() const noexcept
{
	return _data_2;
}

uint64_t info::get_data_3() const noexcept
{
	return _data_3;
}

msg_type info::get_type() const noexcept
{
	return _type;
}

void info::set_info(double data_1, double data_2, uint64_t data_3, msg_type type) noexcept
{
	_data_1 = data_1;
	_data_2 = data_2;
	_data_3 = data_3;
	_type = type;
}

void info::set_data(double data_1, double data_2, uint64_t data_3) noexcept
{
	_data_1 = data_1;
	_data_2 = data_2;
	_data_3 = data_3;
}

void info::set_data_1(double data_1) noexcept
{
	_data_1 = data_1;
}

void info::set_data_2(double data_2) noexcept
{
	_data_2 = data_2;
}

void info::set_data_3(uint64_t data_3) noexcept
{
	_data_3 = data_3;
}

void info::set_type(msg_type type) noexcept
{
	_type = type;
}

info& info::operator=(const info& o) noexcept
{
	_data_1 = o._data_1;
	_data_2 = o._data_2;
	_data_3 = o._data_3;

	_type = o._type;

	return *this;
}

info& info::operator=(info&& o) noexcept
{
	_data_1 = o._data_1;
	_data_2 = o._data_2;
	_data_3 = o._data_3;

	_type = o._type;

	return *this;
}