#pragma once

#include <cstdint>
#include <exception>
#include <initializer_list>
#include <utility>

namespace error_msg
{
	constexpr char nnb_wrong_init_error[] = "NN base wrong initialization";
	constexpr char nnb_empty_error[] = "NN base is empty";

	constexpr char cnn_wrong_init_error[] = "CNN wrong initialization";
	constexpr char cnn_empty_error[] = "CNN is empty";

	constexpr char fnn_wrong_init_error[] = "FNN wrong initialization";
	constexpr char fnn_empty_error[] = "FNN is empty";

	constexpr char convo_wrong_type[] = "Wrong convolution type";

	constexpr char data_sizes_error[] = "Given data are incompatible by size";
	constexpr char tns_sizes_error[] = "Given tensors are incompatible by size";
	constexpr char mtx_sizes_error[] = "Given matrixes are incompatible by size";
	constexpr char vec_sizes_error[] = "Given vectors are incompatible by size";
	constexpr char tns_mtx_sizes_error[] = "Given tensor and matrix are incompatible by size";
	constexpr char tns_vec_sizes_error[] = "Given tensor and vector are incompatible by size";
	constexpr char mtx_vec_sizes_error[] = "Given matrix and vector are incompatible by size";
}

template <typename T, bool initialize = false>
class tns;

template <typename T, bool initialize = false>
class mtx;

template <typename T, bool initialize = false>
class vec;

template <typename T, bool initialize = false>
class data
{
protected:
	T* _data;
	uint64_t _size;

	virtual void assign(const data& o) noexcept
	{
		_data = o._data;
		_size = o._size;
	}

	virtual bool equal(const data& o) const noexcept
	{
		return _size == o._size;
	}

	virtual void zero() noexcept
	{
		_data = nullptr;
		_size = (uint64_t)0;
	}

	data() noexcept : _data(nullptr), _size((uint64_t)0) {}

	data(uint64_t size) : _data(nullptr), _size(size)
	{
		if (_size != (uint64_t)0)
			_data = initialize ? new T[_size]() : new T[_size];
	}

	data(const data& o) : _data(nullptr), _size(o._size)
	{
		if (o._data != nullptr)
		{
			_data = initialize ? new T[_size]() : new T[_size];

			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] = o._data[i];
		}
	}

	data(data&& o) noexcept : _data(o._data), _size(o._size) {}

	data& operator=(const data& o)
	{
		if (_size == o._size)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] = o._data[i];
		else
		{
			if (_data != nullptr)
				delete[] _data;

			if (o._data == nullptr)
				zero();
			else
			{
				assign(o);
				_data = initialize ? new T[_size]() : new T[_size];

				for (uint64_t i = (uint64_t)0; i < _size; ++i)
					_data[i] = o._data[i];
			}
		}

		return *this;
	}

	data& operator=(data&& o) noexcept
	{
		if (_data != nullptr)
			delete[] _data;

		assign(o);
		o.zero();

		return *this;
	}

	data& operator=(const T& sub_o)
	{
		if (_data != nullptr)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] = sub_o;

		return *this;
	}

	data& operator+=(const data& o)
	{
		if (_data == nullptr && o._data == nullptr)
			return *this;
		else if (equal(o))
		{
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] += o._data[i];

			return *this;
		}
		else
			throw std::exception(error_msg::data_sizes_error);
	}

	data& operator-=(const data& o)
	{
		if (_data == nullptr && o._data == nullptr)
			return *this;
		else if (equal(o))
		{
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] -= o._data[i];

			return *this;
		}
		else
			throw std::exception(error_msg::data_sizes_error);
	}

	data& operator*=(const data& o)
	{
		if (_data == nullptr && o._data == nullptr)
			return *this;
		else if (equal(o))
		{
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] *= o._data[i];

			return *this;
		}
		else
			throw std::exception(error_msg::data_sizes_error);
	}

	data& operator/=(const data& o)
	{
		if (_data == nullptr && o._data == nullptr)
			return *this;
		else if (equal(o))
		{
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] /= o._data[i];

			return *this;
		}
		else
			throw std::exception(error_msg::data_sizes_error);
	}

	data& operator+=(const T& sub_o)
	{
		if (_data != nullptr)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] += sub_o;

		return *this;
	}

	data& operator-=(const T& sub_o)
	{
		if (_data != nullptr)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] -= sub_o;

		return *this;
	}

	data& operator*=(const T& sub_o)
	{
		if (_data != nullptr)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] *= sub_o;

		return *this;
	}

	data& operator/=(const T& sub_o)
	{
		if (_data != nullptr)
			for (uint64_t i = (uint64_t)0; i < _size; ++i)
				_data[i] /= sub_o;

		return *this;
	}

public:
	virtual ~data()
	{
		if (_data != nullptr)
			delete[] _data;
	}

	tns<T, initialize> to_tns(uint64_t size_1, uint64_t size_2, uint64_t size_3)
	{
		if (_data == nullptr)
			return tns<T, initialize>(nullptr, (uint64_t)0, (uint64_t)0, (uint64_t)0);
		else if (_size == size_1 * size_2 * size_3)
		{
			tns<T, initialize> result(_data, size_1, size_2, size_3);
			zero();

			return result;
		}
		else
			throw std::exception(error_msg::data_sizes_error);
	}

	mtx<T, initialize> to_mtx(uint64_t size_1, uint64_t size_2)
	{
		if (_data == nullptr)
			return mtx<T, initialize>(nullptr, (uint64_t)0, (uint64_t)0);
		else if (_size == size_1 * size_2)
		{
			mtx<T, initialize> result(_data, size_1, size_2);
			zero();

			return result;
		}
		else
			throw std::exception(error_msg::data_sizes_error);
	}

	vec<T, initialize> to_vec()
	{
		if (_data == nullptr)
			return vec<T, initialize>(nullptr, (uint64_t)0);
		else
		{
			vec<T, initialize> result(_data, _size);
			zero();

			return result;
		}
	}

	uint64_t get_size() const noexcept
	{
		return _size;
	}

	bool is_empty() const noexcept
	{
		return _data == nullptr;
	}

	const T& operator()(uint64_t index) const
	{
		return _data[index];
	}

	T& operator()(uint64_t index)
	{
		return _data[index];
	}

	const T& operator[](uint64_t index) const
	{
		return _data[index];
	}

	T& operator[](uint64_t index)
	{
		return _data[index];
	}
};