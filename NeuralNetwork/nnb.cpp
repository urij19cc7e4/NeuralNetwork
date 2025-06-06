#include "nnb.h"

#include <omp.h>
#include <iomanip>
#include <sstream>

#include "light_appx.h"
#include "rand_sel.h"

using namespace std;
using namespace arithmetic;
using namespace nn_params;

data_set empty_set = { vector<shared_ptr<::data<FLT>>>(), vector<shared_ptr<::data<FLT>>>(), (uint64_t)0 };

inline train_mode operator&(const train_mode& lhs, const train_mode& rhs)
{
	return static_cast<train_mode>(
		static_cast<underlying_type_t<train_mode>>(lhs) &
		static_cast<underlying_type_t<train_mode>>(rhs));
}

inline train_mode operator|(const train_mode& lhs, const train_mode& rhs)
{
	return static_cast<train_mode>(
		static_cast<underlying_type_t<train_mode>>(lhs) |
		static_cast<underlying_type_t<train_mode>>(rhs));
}

inline train_mode operator~(const train_mode& arg)
{
	return static_cast<train_mode>(~static_cast<underlying_type_t<train_mode>>(arg));
}

phase::phase
(
	train_mode mode,
	uint64_t train_batch_size,
	uint64_t test_batch_size,
	uint64_t epoch_count,
	FLT drop_out_rate,
	FLT speed_scale,
	FLT alpha_start,
	FLT alpha_end,
	FLT speed_start,
	FLT speed_end,
	FLT max_error,
	FLT min_error
) noexcept :
	mode(mode),
	train_batch_size(train_batch_size),
	test_batch_size(test_batch_size),
	epoch_count(epoch_count),
	drop_out_rate(drop_out_rate),
	speed_scale(speed_scale),
	alpha_start(alpha_start),
	alpha_end(alpha_end),
	speed_start(speed_start),
	speed_end(speed_end),
	max_error(max_error),
	min_error(min_error)
{}

inline void nnb::add_info(list<info>* value_list, pipe<info>* value_pipe, info&& value)
{
	if (value_list != nullptr)
		value_list->push_back(value);

	if (value_pipe != nullptr)
		value_pipe->push(value);
}

inline bool nnb::check_mode(const train_mode& gen, const train_mode& tar)
{
	return (gen & tar) == tar;
}

list<phase> nnb::get_valid_phases(const list<phase>& phs, uint64_t train_size, uint64_t test_size)
{
	list<phase> valid_phs;

	if (train_size != (uint64_t)0)
		for (phase ph : phs)
		{
			phase valid_ph(ph);

			if (!check_mode(valid_ph.mode, train_mode::BATCH)
				&& !check_mode(valid_ph.mode, train_mode::MINI_BATCH)
				&& !check_mode(valid_ph.mode, train_mode::STOCHASTIC))
				valid_ph.mode = valid_ph.mode | train_mode::STOCHASTIC;

			if (check_mode(valid_ph.mode, train_mode::BATCH))
			{
				valid_ph.train_batch_size = train_size;
				valid_ph.test_batch_size = test_size;
			}
			else
			{
				if (valid_ph.train_batch_size > train_size)
					valid_ph.train_batch_size = train_size;

				if (valid_ph.test_batch_size > test_size)
					valid_ph.test_batch_size = test_size;
			}

			if (valid_ph.train_batch_size != (uint64_t)0)
			{
				if (!check_mode(valid_ph.mode, train_mode::RANDOM_TRAIN_SELECTION)
					&& !check_mode(valid_ph.mode, train_mode::STRICT_TRAIN_SELECTION))
					if (train_size > (uint64_t)250)
						valid_ph.mode = valid_ph.mode | train_mode::RANDOM_TRAIN_SELECTION;
					else
						valid_ph.mode = valid_ph.mode | train_mode::STRICT_TRAIN_SELECTION;

				if (!check_mode(valid_ph.mode, train_mode::RANDOM_TEST_SELECTION)
					&& !check_mode(valid_ph.mode, train_mode::STRICT_TEST_SELECTION))
					if (test_size > (uint64_t)250)
						valid_ph.mode = valid_ph.mode | train_mode::RANDOM_TEST_SELECTION;
					else
						valid_ph.mode = valid_ph.mode | train_mode::STRICT_TEST_SELECTION;

				if (valid_ph.test_batch_size == (uint64_t)0 || test_size == (uint64_t)0)
					valid_ph.mode = valid_ph.mode & (~train_mode::CROSS_TEST);

				if (valid_ph.drop_out_rate == (FLT)0)
					valid_ph.mode = valid_ph.mode & (~train_mode::DROP_OUT);

				valid_phs.push_back(move(valid_ph));
			}
		}

	return valid_phs;
}

string nnb::get_header(uint64_t layers, uint64_t params)
{
	ostringstream header;

	header << "Neural Network Binary Train Data";
	header << "\r\nLayer Count: " << setw(20) << right << layers;
	header << "\r\nParam Count: " << setw(20) << right << params;
	header << "\r\n";

	return header.str();
}

nnb::nnb() noexcept : _lays(nullptr), _size((uint64_t)0) {}

nnb::nnb(initializer_list<unique_ptr<nn_info>> lays_info) : _lays(nullptr), _size(lays_info.size())
{
	if (_size != (uint64_t)0)
	{
		_lays = new nn * [_size];
		const unique_ptr<nn_info>* elems = lays_info.begin();

		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			_lays[i] = elems[i]->create_new();
	}
}

nnb::nnb(ifstream& file) : nnb()
{
	string header = get_header(_size, get_param_count());
	char* temp = new char[header.size()];

	file.read(reinterpret_cast<char*>(temp), sizeof(header.c_str()[(uint64_t)0]) * (uint64_t)header.size());
	file.read(reinterpret_cast<char*>(&_size), sizeof(_size));

	if (_size != (uint64_t)0)
	{
		_lays = new nn * [_size];

		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			_lays[i] = nn::create_from_file(file);
	}

	delete[] temp;
}

nnb::nnb(const nnb& o) : _lays(nullptr), _size(o._size)
{
	if (_size != (uint64_t)0)
	{
		_lays = new nn * [_size];

		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			_lays[i] = o._lays[i]->create_new();
	}
}

nnb::nnb(nnb&& o) noexcept : _lays(o._lays), _size(o._size)
{
	o._lays = nullptr;
	o._size = (uint64_t)0;
}

nnb::~nnb()
{
	if (_lays != nullptr)
	{
		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			delete _lays[i];

		delete[] _lays;
	}
}

void nnb::save_to_file(ofstream& file) const
{
	string header = get_header(_size, get_param_count());

	file.write(reinterpret_cast<const char*>(header.c_str()), sizeof(header.c_str()[(uint64_t)0]) * (uint64_t)header.size());
	file.write(reinterpret_cast<const char*>(&_size), sizeof(_size));

	for (uint64_t i = (uint64_t)0; i < _size; ++i)
		_lays[i]->save_to_file(file);
}

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

::data<FLT>* nnb::pass(const ::data<FLT>& _data) const
{
	if (is_empty())
		throw exception(error_msg::nnb_empty_error);
	else
	{
		::data<FLT>* result_ptr = _lays[(uint64_t)0]->pass_fwd(_data);

		for (uint64_t i = (uint64_t)1; i < _size; ++i)
		{
			::data<FLT>* temp = _lays[i]->pass_fwd(*result_ptr);
			delete result_ptr;
			result_ptr = temp;
		}

		return result_ptr;
	}
}

void nnb::train
(
	data_set train_set,
	data_set test_set,
	train_mode mode,
	const std::list<phase>* phase_list,
	std::list<info>* error_list,
	pipe<info>* error_pipe,
	uint64_t train_batch_size,
	uint64_t test_batch_size,
	uint64_t epoch_count,
	uint64_t fail_count,
	uint64_t test_freq,
	FLT drop_out_rate,
	FLT speed_scale,
	FLT alpha_start,
	FLT alpha_end,
	FLT speed_start,
	FLT speed_end,
	FLT max_error,
	FLT min_error
)
{
	if (is_empty())
		throw exception(error_msg::nnb_empty_error);
	else if (train_set.size == (uint64_t)0)
		throw exception("...");
	else
	{
		list<phase> temp_phase_list;

		if (phase_list == nullptr)
		{
			temp_phase_list.push_back(move(phase(
				mode,
				train_batch_size,
				test_batch_size,
				epoch_count,
				drop_out_rate,
				speed_scale,
				alpha_start,
				alpha_end,
				speed_start,
				speed_end,
				max_error,
				min_error
			)));

			temp_phase_list = move(get_valid_phases(temp_phase_list, train_set.size, test_set.size));
		}
		else
			temp_phase_list = move(get_valid_phases(*phase_list, train_set.size, test_set.size));

		if (temp_phase_list.size() == (uint64_t)0)
			throw exception("...");
		else
		{
			bool max_epo_reached = true;
			epoch_count = (uint64_t)0;
			for (phase ph : temp_phase_list)
				epoch_count += ph.epoch_count;

			add_info(error_list, error_pipe, move(info(epoch_count, msg_type::count_epo)));
			epoch_count = (uint64_t)0;

			uint64_t last_layer = _size - (uint64_t)1;
			uint64_t test_fails = (uint64_t)0;

			FLT min_test_error = (FLT)0;
			FLT prev_test_error = (FLT)0;

			for (phase ph : temp_phase_list)
			{
				bool quit = false;

				light_appx alpha_appx(ph.alpha_start, ph.alpha_end, ph.epoch_count);
				light_appx speed_appx(ph.speed_start, ph.speed_end, ph.epoch_count);

				unique_ptr<FLT[]> speed_scale_vals(new FLT[_size]);
				ph.speed_scale = max(ph.speed_scale * (FLT)_size, (FLT)1);

				unique_ptr<rand_sel_i<uint64_t>>train_set_randomizer = check_mode(ph.mode, train_mode::RANDOM_TRAIN_SELECTION)
					? unique_ptr<rand_sel_i<uint64_t>>((new rand_sel<uint64_t, false>(train_set.size - (uint64_t)1, (uint64_t)0)))
					: unique_ptr<rand_sel_i<uint64_t>>((new rand_sel<uint64_t, true>(train_set.size - (uint64_t)1, (uint64_t)0)));
				unique_ptr<rand_sel_i<uint64_t>>test_set_randomizer = check_mode(ph.mode, train_mode::RANDOM_TEST_SELECTION)
					? unique_ptr<rand_sel_i<uint64_t>>((new rand_sel<uint64_t, false>(test_set.size - (uint64_t)1, (uint64_t)0)))
					: unique_ptr<rand_sel_i<uint64_t>>((new rand_sel<uint64_t, true>(test_set.size - (uint64_t)1, (uint64_t)0)));

				if (check_mode(ph.mode, train_mode::BATCH) || check_mode(ph.mode, train_mode::MINI_BATCH))
				{
					uint64_t copy_count = min((uint64_t)omp_get_max_threads(), ph.train_batch_size);
					uint64_t cycle_count = (ph.train_batch_size + copy_count - (uint64_t)1) / copy_count;
					copy_count = (ph.train_batch_size + cycle_count - (uint64_t)1) / cycle_count;

					unique_ptr<unique_ptr<nn_trainy>[]> train_data(new unique_ptr<nn_trainy>[_size * copy_count]);
					unique_ptr<unique_ptr<nn_trainy_batch>[]> train_base(new unique_ptr<nn_trainy_batch>[_size]);
					unique_ptr<uint64_t[]> set_nums(new uint64_t[copy_count]);

					for (uint64_t i = (uint64_t)0, j = (uint64_t)0; i < copy_count; ++i)
					{
						{
							double drop_out_rate = check_mode(ph.mode, train_mode::DROP_OUT) && (uint64_t)0 != last_layer
								? (double)ph.drop_out_rate : (double)0;

							train_data[j] = move(unique_ptr<nn_trainy>(
								_lays[(uint64_t)0]->get_trainy(*(train_set.idata[(uint64_t)0]), drop_out_rate, false)
							));

							++j;
						}
						for (uint64_t k = (uint64_t)1; k < _size; ++j, ++k)
						{
							double drop_out_rate = check_mode(ph.mode, train_mode::DROP_OUT) && k != last_layer
								? (double)ph.drop_out_rate : (double)0;

							train_data[j] = move(unique_ptr<nn_trainy>(
								_lays[k]->get_trainy(*(train_data[j - (uint64_t)1]), drop_out_rate, false)
							));
						}
					}

					train_base[(uint64_t)0] = move(unique_ptr<nn_trainy_batch>(
						_lays[(uint64_t)0]->get_trainy_batch(*(train_set.idata[(uint64_t)0]))
					));
					for (uint64_t i = (uint64_t)1; i < _size; ++i)
						train_base[i] = move(unique_ptr<nn_trainy_batch>(
							_lays[i]->get_trainy_batch(*(train_data[i - (uint64_t)1]))
						));

					for (uint64_t i = (uint64_t)0; i < ph.epoch_count; ++i, ++epoch_count)
					{
						FLT train_err = (FLT)0, test_err = (FLT)0;
						FLT alpha_epo = alpha_appx.forward(), speed_epo = speed_appx.forward();
						light_appx speed_scale_appx(ph.speed_scale * speed_epo, speed_epo, _size, (FLT)0, (FLT)2);

						for (uint64_t j = (uint64_t)0; j < _size; ++j)
							speed_scale_vals[j] = speed_scale_appx.forward();

					#pragma omp parallel for num_threads(_size)
						for (int64_t omp_thread = (int64_t)0; omp_thread < (int64_t)_size; ++omp_thread)
							train_base[(uint64_t)omp_thread]->begin_update(alpha_epo);

						for (uint64_t j = (uint64_t)0; j < cycle_count; ++j)
						{
							uint64_t copy_left = j == cycle_count - (uint64_t)1 ? ph.train_batch_size - j * copy_count : copy_count;

							for (uint64_t k = (uint64_t)0; k < copy_left; ++k)
								set_nums[k] = train_set_randomizer->next();

						#pragma omp parallel for reduction(+:train_err) num_threads(copy_left)
							for (int64_t omp_thread = (int64_t)0; omp_thread < (int64_t)copy_left; ++omp_thread)
							{
								uint64_t num = set_nums[(uint64_t)omp_thread];
								uint64_t data_num = (uint64_t)omp_thread * _size;

								_lays[(uint64_t)0]->train_fwd(*(train_data[data_num]), *(train_set.idata[num]));
								for (uint64_t k = (uint64_t)1; k < _size; ++k)
									_lays[k]->train_fwd(*(train_data[data_num + k]), *(train_data[data_num + k - (uint64_t)1]));

								train_err += _lays[last_layer]->train_bwd(*(train_data[data_num + last_layer]), *(train_set.odata[num]));
								for (uint64_t k = last_layer; k > (uint64_t)0; --k)
									_lays[k]->train_bwd(*(train_data[data_num + k]), *(train_data[data_num + k - (uint64_t)1]));
							}

						#pragma omp parallel for num_threads(_size)
							for (int64_t omp_thread = (int64_t)0; omp_thread < (int64_t)_size; ++omp_thread)
							{
								FLT speed_lay = speed_scale_vals[(uint64_t)omp_thread] / (FLT)copy_left;

								if (omp_thread == (int64_t)0)
									for (uint64_t i = (uint64_t)0, j = (uint64_t)omp_thread; i < copy_left; ++i, j += _size)
										train_base[(uint64_t)omp_thread]->update(
											*(train_data[j]), *(train_set.idata[set_nums[i]]), speed_lay
										);
								else
									for (uint64_t i = (uint64_t)0, j = (uint64_t)omp_thread; i < copy_left; ++i, j += _size)
										train_base[(uint64_t)omp_thread]->update(
											*(train_data[j]), *(train_data[j - (uint64_t)1]), speed_lay
										);
							}
						}

					#pragma omp parallel for num_threads(_size)
						for (int64_t omp_thread = (int64_t)0; omp_thread < (int64_t)_size; ++omp_thread)
							_lays[(uint64_t)omp_thread]->train_upd(*(train_base[(uint64_t)omp_thread]));

						train_err /= (FLT)ph.train_batch_size;

						if (train_err < ph.min_error)
						{
							add_info(error_list, error_pipe, move(info(epoch_count + (uint64_t)1, msg_type::min_err_reached)));
							max_epo_reached = false;
							quit = true;

							break;
						}

						if (check_mode(ph.mode, train_mode::CROSS_TEST))
						{
							if (epoch_count % test_freq == (uint64_t)0)
							{
							#pragma omp parallel for reduction(+:test_err) num_threads(copy_count)
								for (int64_t omp_thread = (int64_t)0; omp_thread < (int64_t)ph.test_batch_size; ++omp_thread)
								{
									uint64_t num;

								#pragma omp critical
									{
										num = test_set_randomizer->next();
									}

									unique_ptr<::data<FLT>> fwd_result(pass(*(test_set.idata[num])));

									for (uint64_t k = (uint64_t)0; k < fwd_result->get_size(); ++k)
									{
										FLT loss = (*(test_set.odata[num]))(k) - (*fwd_result)(k);
										test_err += loss * loss;
									}
								}

								test_err /= (FLT)ph.test_batch_size;
								prev_test_error = test_err;
							}
							else
								test_err = prev_test_error;

							if (epoch_count == (uint64_t)0 || min_test_error > test_err)
								min_test_error = test_err;

							if (test_err > min_test_error * (ph.max_error + (FLT)1))
								++test_fails;

							if (test_err > min_test_error * (ph.max_error * (FLT)2 + (FLT)1) || test_fails > fail_count)
							{
								add_info(error_list, error_pipe, move(info(epoch_count + (uint64_t)1, msg_type::max_err_reached)));
								max_epo_reached = false;
								quit = true;

								break;
							}
						}

						add_info(error_list, error_pipe, move(info(train_err, test_err)));
					}
				}
				else if (check_mode(ph.mode, train_mode::STOCHASTIC))
				{
					unique_ptr<unique_ptr<nn_trainy>[]> train_data(new unique_ptr<nn_trainy>[_size]);

					{
						double drop_out_rate = check_mode(ph.mode, train_mode::DROP_OUT) && (uint64_t)0 != last_layer
							? (double)ph.drop_out_rate : (double)0;

						train_data[(uint64_t)0] = move(unique_ptr<nn_trainy>(
							_lays[(uint64_t)0]->get_trainy(*(train_set.idata[(uint64_t)0]), drop_out_rate, true)
						));
					}
					for (uint64_t i = (uint64_t)1; i < _size; ++i)
					{
						double drop_out_rate = check_mode(ph.mode, train_mode::DROP_OUT) && i != last_layer
							? (double)ph.drop_out_rate : (double)0;

						train_data[i] = move(unique_ptr<nn_trainy>(
							_lays[i]->get_trainy(*(train_data[i - (uint64_t)1]), drop_out_rate, true)
						));
					}

					for (uint64_t i = (uint64_t)0; i < ph.epoch_count; ++i, ++epoch_count)
					{
						FLT train_err = (FLT)0, test_err = (FLT)0;
						FLT alpha_epo = alpha_appx.forward(), speed_epo = speed_appx.forward();
						light_appx speed_scale_appx(ph.speed_scale * speed_epo, speed_epo, _size, (FLT)0, (FLT)2);

						for (uint64_t j = (uint64_t)0; j < _size; ++j)
							speed_scale_vals[j] = speed_scale_appx.forward();

						for (uint64_t j = (uint64_t)0; j < ph.train_batch_size; ++j)
						{
							uint64_t num = train_set_randomizer->next();

							_lays[(uint64_t)0]->train_fwd(*(train_data[(uint64_t)0]), *(train_set.idata[num]));
							for (uint64_t k = (uint64_t)1; k < _size; ++k)
								_lays[k]->train_fwd(*(train_data[k]), *(train_data[k - (uint64_t)1]));

							train_err += _lays[last_layer]->train_bwd(*(train_data[last_layer]), *(train_set.odata[num]));
							for (uint64_t k = last_layer; k > (uint64_t)0; --k)
								_lays[k]->train_bwd(*(train_data[k]), *(train_data[k - (uint64_t)1]));

							{
								train_data[(uint64_t)0]->update(*(train_set.idata[num]), alpha_epo, speed_scale_vals[(uint64_t)0]);
								_lays[(uint64_t)0]->train_upd(*(train_data[(uint64_t)0]));
							}
							for (uint64_t k = (uint64_t)1; k < _size; ++k)
							{
								train_data[k]->update(*(train_data[k - (uint64_t)1]), alpha_epo, speed_scale_vals[k]);
								_lays[k]->train_upd(*(train_data[k]));
							}
						}

						train_err /= (FLT)ph.train_batch_size;

						if (train_err < ph.min_error)
						{
							add_info(error_list, error_pipe, move(info(epoch_count + (uint64_t)1, msg_type::min_err_reached)));
							max_epo_reached = false;
							quit = true;

							break;
						}

						if (check_mode(ph.mode, train_mode::CROSS_TEST))
						{
							if (epoch_count % test_freq == (uint64_t)0)
							{
								for (uint64_t j = (uint64_t)0; j < ph.test_batch_size; ++j)
								{
									uint64_t num = test_set_randomizer->next();
									unique_ptr<::data<FLT>> fwd_result(pass(*(test_set.idata[num])));

									for (uint64_t k = (uint64_t)0; k < fwd_result->get_size(); ++k)
									{
										FLT loss = (*(test_set.odata[num]))(k) - (*fwd_result)(k);
										test_err += loss * loss;
									}
								}

								test_err /= (FLT)ph.test_batch_size;
								prev_test_error = test_err;
							}
							else
								test_err = prev_test_error;

							if (epoch_count == (uint64_t)0 || min_test_error > test_err)
								min_test_error = test_err;

							if (test_err > min_test_error * (ph.max_error + (FLT)1))
								++test_fails;

							if (test_err > min_test_error * (ph.max_error * (FLT)2 + (FLT)1) || test_fails > fail_count)
							{
								add_info(error_list, error_pipe, move(info(epoch_count + (uint64_t)1, msg_type::max_err_reached)));
								max_epo_reached = false;
								quit = true;

								break;
							}
						}

						add_info(error_list, error_pipe, move(info(train_err, test_err)));
					}
				}
				else
					throw exception("...");

				if (quit)
					break;
			}

			if (max_epo_reached)
				add_info(error_list, error_pipe, move(info(msg_type::max_epo_reached)));
		}
	}
}

nnb& nnb::operator=(const nnb& o)
{
	if (_lays != nullptr)
	{
		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			delete _lays[i];

		delete[] _lays;
	}

	if (o._lays == nullptr)
	{
		_lays = nullptr;
		_size = (uint64_t)0;
	}
	else
	{
		_lays = new nn * [o._size];
		_size = o._size;

		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			_lays[i] = o._lays[i]->create_new();
	}

	return *this;
}

nnb& nnb::operator=(nnb&& o) noexcept
{
	if (_lays != nullptr)
	{
		for (uint64_t i = (uint64_t)0; i < _size; ++i)
			delete _lays[i];

		delete[] _lays;
	}

	_lays = o._lays;
	_size = o._size;

	o._lays = nullptr;
	o._size = (uint64_t)0;

	return *this;
}