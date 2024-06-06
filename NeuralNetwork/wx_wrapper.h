#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <thread>

#include "info.h"
#include "pipe.h"

class wx_wrapper
{
private:
	static constexpr char _wnd_name[] = "Neural Network Train Graph";

	struct wnd_info
	{
		std::thread thread;
		void* wx_wnd;
	};

	static inline uint64_t _count = (uint64_t)0;
	static inline std::mutex _mutex = std::mutex();
	static inline std::list<wnd_info> _procs = std::list<wnd_info>();

	static void wnd_proc(void* graph_wnd) noexcept;

public:
	wx_wrapper();
	wx_wrapper(int argc, char* argv[]);
	wx_wrapper(const wx_wrapper& o) = delete;
	wx_wrapper(wx_wrapper&& o) = delete;
	~wx_wrapper();

	void create_wnd(const std::list<info>& data, pipe<info>& data_pipe, const std::string& name = _wnd_name) const;
	void create_wnd(const std::list<info>& data, const std::string& name = _wnd_name) const;
	void create_wnd(pipe<info>& data_pipe, const std::string& name = _wnd_name) const;
};