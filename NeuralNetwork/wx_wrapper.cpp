#include "wx_wrapper.h"
#include "window.h"

#include <exception>
#include <utility>

using namespace std;
using namespace window;

void wx_wrapper::wnd_proc() noexcept
{
	try
	{
		wxGetApp().OnInit();
		wxGetApp().OnRun();
		wxGetApp().OnExit();
	}
	catch (...) {}
}

wx_wrapper::wx_wrapper() : wx_wrapper(0, nullptr) {}

wx_wrapper::wx_wrapper(int argc, char* argv[])
{
	lock_guard<mutex> lock(_mutex);

	if (_count == (uint64_t)0)
	{
		if (!wxEntryStart(argc, argv))
			throw exception("Error starting wxWidgets");

		_thread = thread(wnd_proc);
	}

	++_count;
}

wx_wrapper::~wx_wrapper()
{
	lock_guard<mutex> lock(_mutex);

	--_count;

	if (_count == (uint64_t)0)
	{
		for (wxWindow* wx_wnd_frm = wxGetApp().GetMainTopWindow(); wx_wnd_frm != nullptr; wx_wnd_frm = wxGetApp().GetMainTopWindow())
			PostMessage((HWND)wx_wnd_frm->GetHandle(), (UINT)WM_CLOSE, (WPARAM)0, (LPARAM)0);

		if (_thread.joinable())
			_thread.join();

		_thread = thread();
		wxEntryCleanup();
	}
}

void wx_wrapper::create_graph_wnd(const list<info>& data, pipe<info>& data_pipe, const string& name) const
{
	lock_guard<mutex> lock(_mutex);

	wxGetApp().CallAfter([_data = data, _data_pipe = &data_pipe, _name = name]() mutable noexcept
		{
			try
			{
				(new GraphFrame(move(_data), _data_pipe, move(_name)))->Show();
			}
			catch (...) {}
		});
}

void wx_wrapper::create_graph_wnd(const list<info>& data, const string& name) const
{
	lock_guard<mutex> lock(_mutex);

	wxGetApp().CallAfter([_data = data, _name = name]() mutable noexcept
		{
			try
			{
				(new GraphFrame(move(_data), nullptr, move(_name)))->Show();
			}
			catch (...) {}
		});
}

void wx_wrapper::create_graph_wnd(pipe<info>& data_pipe, const string& name) const
{
	lock_guard<mutex> lock(_mutex);

	wxGetApp().CallAfter([_data_pipe = &data_pipe, _name = name]() mutable noexcept
		{
			try
			{
				(new GraphFrame(move(list<info>()), _data_pipe, move(_name)))->Show();
			}
			catch (...) {}
		});
}

void wx_wrapper::create_nn_io_wnd() const
{
	lock_guard<mutex> lock(_mutex);

	wxGetApp().CallAfter([]() mutable noexcept
		{
			try
			{
				(new NNIOFrame())->Show();
			}
			catch (...) {}
		});
}