#pragma once

#include <list>
#include <mutex>
#include <utility>

template <typename T>
class pipe
{
private:
	std::mutex _mutex;
	std::list<T> _queue;

public:
	pipe() {}

	pipe(const pipe& o) = delete;

	pipe(pipe&& o) = delete;

	~pipe() {}

	bool is_empty() const noexcept
	{
		std::lock_guard<std::mutex> lock(_mutex);
		return _queue.empty();
	}

	void pop(T& value)
	{
		std::lock_guard<std::mutex> lock(_mutex);

		if (!_queue.empty())
		{
			value = std::move(_queue.front());
			_queue.pop_front();
		}
	}

	void push(const T& value)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		_queue.push_back(value);
	}

	void push(T&& value)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		_queue.push_back(std::move(value));
	}
};