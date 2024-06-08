#include "wx_wrapper.h"
#include "window.h"

#include <exception>
#include <utility>

using namespace std;
using namespace window;

void wx_wrapper::wnd_proc(GraphWnd* graph_wnd) noexcept
{
	try
	{
		if (graph_wnd != nullptr)
		{
			graph_wnd->OnInit();
			graph_wnd->OnRun();
			graph_wnd->OnExit();
		}
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

		_window = new GraphWnd();
		_thread = thread(wnd_proc, _window);
	}

	++_count;
}

wx_wrapper::~wx_wrapper()
{
	lock_guard<mutex> lock(_mutex);

	--_count;

	if (_count == (uint64_t)0)
	{
		for (wxWindow* wx_wnd_frm = _window->GetMainTopWindow(); wx_wnd_frm != nullptr; wx_wnd_frm = _window->GetMainTopWindow())
			PostMessage((HWND)wx_wnd_frm->GetHandle(), (UINT)WM_CLOSE, (WPARAM)0, (LPARAM)0);

		if (_thread.joinable())
			_thread.join();

		delete _window;
		_window = nullptr;
		_thread = thread();

		wxEntryCleanup();
	}
}

void wx_wrapper::create_wnd(const list<info>& data, pipe<info>& data_pipe, const string& name) const
{
	lock_guard<mutex> lock(_mutex);
	_window->CallAfter([=, &data_pipe] { wxGetApp().NewFrame(data, &data_pipe, name); });
}

void wx_wrapper::create_wnd(const list<info>& data, const string& name) const
{
	lock_guard<mutex> lock(_mutex);
	_window->CallAfter([=] { wxGetApp().NewFrame(data, nullptr, name); });
}

void wx_wrapper::create_wnd(pipe<info>& data_pipe, const string& name) const
{
	lock_guard<mutex> lock(_mutex);
	_window->CallAfter([=, &data_pipe] { wxGetApp().NewFrame(list<info>(), &data_pipe, name); });
}