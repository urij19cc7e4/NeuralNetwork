#include "window.h"
#include "mathplot.cpp"

using namespace std;
using namespace window;

void window::GraphWnd::Init(const list<info>* data, pipe<info>* data_pipe, const string* name)
{
	if (data != nullptr)
		_data = *data;

	if (data_pipe != nullptr)
		_data_pipe = data_pipe;

	if (name != nullptr)
		_name = *name;
}

bool window::GraphWnd::OnInit()
{
	GraphFrame* wnd_frame = new GraphFrame(move(_data), _data_pipe, move(_name));
	wnd_frame->Show();

	return true;
}

int window::GraphWnd::OnExit()
{
	return 0;
}

void window::GraphFrame::AdderProc() noexcept
{
	try
	{
		while (adderProcRun)
		{
			if (!dataPipe->is_empty())
			{
				while (!dataPipe->is_empty())
				{
					info data;
					dataPipe->pop(data);

					ProcInfo(data);
				}

				plotterWnd.Fit();
			}

			this_thread::sleep_for(chrono::milliseconds((uint64_t)1));
		}
	}
	catch (...) {}
}

void window::GraphFrame::ProcInfo(const info& data)
{
	lock_guard<mutex> lock(adderMutex);

	switch (data.get_type())
	{
	case msg_type::data:

		trainErr->ReplaceDataByX((double)count, data.get_data_1());
		testErr->ReplaceDataByX((double)count, data.get_data_2());
		++count;

		break;

	case msg_type::count_epo:

		trainErr->Reserve(data.get_data_3());
		testErr->Reserve(data.get_data_3());

		for (uint64_t i = count; i < data.get_data_3(); ++i)
		{
			trainErr->AddData((double)i, (double)0);
			testErr->AddData((double)i, (double)0);
		}

		break;

	case msg_type::count_set_1:

		trainSize = data.get_data_3();

		break;

	case msg_type::count_set_2:

		testSize = data.get_data_3();

		break;

	case msg_type::batch_mode:

		modeName += "Batch Parallel";

		break;

	case msg_type::stoch_mode:

		modeName += "Stochatic Seq";

		break;

	case msg_type::cross_mode:

		modeName += " with Cross Testing";

		break;

	case msg_type::max_epo_reached:

		adderProcRun = false;
		resultName = "Maximum Epoch Count reached";

		break;

	case msg_type::max_err_reached:

		adderProcRun = false;
		resultName = "Maximum Test Error reached for " + to_string(data.get_data_3()) + " epoch(s)";

		break;

	case msg_type::min_err_reached:

		adderProcRun = false;
		resultName = "Minimum Train Error reached for " + to_string(data.get_data_3()) + " epoch(s)";

		break;

	default:
		break;
	}

	UpdateStatusBar();
}

void window::GraphFrame::UpdateStatusBar()
{
	wxGetApp().CallAfter([this]
		{
			lock_guard<mutex> lock(this->adderMutex);

			this->SetStatusText(_((string("Train Mode: ") + this->modeName).c_str()), 0);
			this->SetStatusText(_((string("Train Set: ") + to_string(this->trainSize)).c_str()), 1);
			this->SetStatusText(_((string("Test Set: ") + to_string(this->testSize)).c_str()), 2);
			this->SetStatusText(_(this->resultName.c_str()), 3);
		});
}

window::GraphFrame::GraphFrame(list<info>&& data, pipe<info>* data_pipe, string&& name)
	: wxFrame(nullptr, wxID_ANY, _(name.c_str()), wxDefaultPosition, wxSize(800, 400),
		wxCAPTION | wxCLIP_CHILDREN | wxCLOSE_BOX | wxSYSTEM_MENU | wxMINIMIZE_BOX
		| wxMAXIMIZE_BOX | wxRESIZE_BORDER, _(name.c_str())),
	adderMutex(),
	adderProc(),
	adderProcRun(true),
	count((uint64_t)0),
	dataPipe(data_pipe),
	axisX(new mpScaleX(_("Epoch"))),
	axisY(new mpScaleY(_("Error"))),
	infoLegend(new mpInfoLegend(wxRect(640, 10, 125, 40))),
	trainErr(new mpFXYVector(_("Train Set Error"))),
	testErr(new mpFXYVector(_("Test Set Error"))),
	plotterWnd(this, wxID_ANY, wxPoint(0, 0), wxSize(800, 400), wxSUNKEN_BORDER),
	modeName(),
	resultName("Processing..."),
	trainSize((uint64_t)0),
	testSize((uint64_t)0)
{
	wxFont font = wxFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_SLANT, wxFONTWEIGHT_SEMIBOLD);

	axisX->SetFont(font);
	axisY->SetFont(font);

	axisX->SetDrawOutsideMargins(false);
	axisY->SetDrawOutsideMargins(false);

	infoLegend->SetFont(font);
	infoLegend->SetItemMode(mpLEGEND_SQUARE);

	trainErr->SetPen(wxPen(wxColour((wxColourBase::ChannelType)255, (wxColourBase::ChannelType)0,
		(wxColourBase::ChannelType)0, (wxColourBase::ChannelType)255), 3, wxPenStyle::wxPENSTYLE_DOT_DASH));
	testErr->SetPen(wxPen(wxColour((wxColourBase::ChannelType)0, (wxColourBase::ChannelType)255,
		(wxColourBase::ChannelType)0, (wxColourBase::ChannelType)255), 3, wxPenStyle::wxPENSTYLE_DOT_DASH));

	trainErr->SetContinuity(true);
	testErr->SetContinuity(true);

	trainErr->SetDrawOutsideMargins(false);
	testErr->SetDrawOutsideMargins(false);

	trainErr->ShowName(false);
	testErr->ShowName(false);

	for (list<info>::iterator i = data.begin(); i != data.end(); ++i)
		ProcInfo(*i);

	plotterWnd.AddLayer(axisX);
	plotterWnd.AddLayer(axisY);
	plotterWnd.AddLayer(infoLegend);
	plotterWnd.AddLayer(trainErr);
	plotterWnd.AddLayer(testErr);

	plotterWnd.EnableDoubleBuffer(true);
	plotterWnd.EnableMousePanZoom(false);
	plotterWnd.EnableTouchEvents(0);

	plotterWnd.SetAutoLayout(true);
	plotterWnd.SetMargins(25, 50, 25, 50);

	plotterWnd.Fit();

	adderProc = thread([this] { this->AdderProc(); });

	CreateStatusBar(4);
	UpdateStatusBar();
}

window::GraphFrame::~GraphFrame()
{
	adderProcRun = false;

	if (adderProc.joinable())
		adderProc.join();
}

namespace window
{
	wxIMPLEMENT_APP_NO_MAIN(GraphWnd);
}