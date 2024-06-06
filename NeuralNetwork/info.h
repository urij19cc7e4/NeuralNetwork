#pragma once

#include <cstdint>

enum class msg_type : uint64_t
{
	data,
	count_epo,
	count_set_1,
	count_set_2,
	batch_mode,
	stoch_mode,
	cross_mode,
	max_epo_reached,
	max_err_reached,
	min_err_reached,
	__count__
};

class info
{
private:
	double _data_1;
	double _data_2;
	uint64_t _data_3;
	msg_type _type;

public:
	info() noexcept;
	info(double data_1, double data_2, uint64_t data_3, msg_type type) noexcept;
	info(double data_1, double data_2, uint64_t data_3) noexcept;
	info(double data_1, double data_2, msg_type type) noexcept;
	info(double data_1, double data_2) noexcept;
	info(uint64_t data_3, msg_type type) noexcept;
	info(uint64_t data_3) noexcept;
	info(msg_type type) noexcept;
	info(const info& o) noexcept;
	info(info&& o) noexcept;
	~info() noexcept;

	double get_data_1() const noexcept;
	double get_data_2() const noexcept;
	uint64_t get_data_3() const noexcept;
	msg_type get_type() const noexcept;

	void set_info(double data_1, double data_2, uint64_t data_3, msg_type type) noexcept;
	void set_data(double data_1, double data_2, uint64_t data_3) noexcept;
	void set_data_1(double data_1) noexcept;
	void set_data_2(double data_2) noexcept;
	void set_data_3(uint64_t data_3) noexcept;
	void set_type(msg_type type) noexcept;

	info& operator=(const info& o) noexcept;
	info& operator=(info&& o) noexcept;
};