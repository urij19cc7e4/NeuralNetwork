#include "wx_wrapper.h"
#include "window.h"

#include <exception>
#include <utility>

using namespace std;
using namespace window;

void wx_wrapper::wnd_proc(void* graph_wnd) noexcept
{
	try
	{
		if (graph_wnd != nullptr)
		{
			((GraphWnd*)graph_wnd)->OnInit();
			((GraphWnd*)graph_wnd)->OnRun();
			((GraphWnd*)graph_wnd)->OnExit();

			delete (GraphWnd*)graph_wnd;
		}
	}
	catch (...) {}
}

wx_wrapper::wx_wrapper() : wx_wrapper(0, nullptr) {}

wx_wrapper::wx_wrapper(int argc, char* argv[])
{
	lock_guard<mutex> lock(_mutex);

	if (_count == (uint64_t)0 && !wxEntryStart(argc, argv))
		throw exception("Error starting wxWidgets");

	++_count;
}

wx_wrapper::~wx_wrapper()
{
	lock_guard<mutex> lock(_mutex);

	--_count;

	if (_count == (uint64_t)0)
	{
		for (list<wnd_info>::iterator i = _procs.begin(); i != _procs.end(); ++i)
		{
			wxWindow* wx_wnd_obj = ((GraphWnd*)i->wx_wnd)->GetMainTopWindow();

			if (wx_wnd_obj != nullptr)
				PostMessage((HWND)wx_wnd_obj->GetHandle(), (UINT)WM_CLOSE, (WPARAM)0, (LPARAM)0);
		}

		for (list<wnd_info>::iterator i = _procs.begin(); i != _procs.end(); ++i)
			if (i->thread.joinable())
				i->thread.join();

		_procs.clear();
		wxEntryCleanup();
	}
}

void wx_wrapper::create_wnd(const list<info>& data, pipe<info>& data_pipe, const string& name) const
{
	lock_guard<mutex> lock(_mutex);

	GraphWnd* graph_wnd = new GraphWnd();
	graph_wnd->Init(&data, &data_pipe, &name);

	_procs.push_back({ move(thread(wnd_proc, graph_wnd)), graph_wnd });
}

void wx_wrapper::create_wnd(const list<info>& data, const string& name) const
{
	lock_guard<mutex> lock(_mutex);

	GraphWnd* graph_wnd = new GraphWnd();
	graph_wnd->Init(&data, nullptr, &name);

	_procs.push_back({ move(thread(wnd_proc, graph_wnd)), graph_wnd });
}

void wx_wrapper::create_wnd(pipe<info>& data_pipe, const string& name) const
{
	lock_guard<mutex> lock(_mutex);

	GraphWnd* graph_wnd = new GraphWnd();
	graph_wnd->Init(nullptr, &data_pipe, &name);

	_procs.push_back({ move(thread(wnd_proc, graph_wnd)), graph_wnd });
}