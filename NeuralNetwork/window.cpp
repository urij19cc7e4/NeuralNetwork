#include "window.h"
#include "mathplot.cpp"

#include <chrono>
#include <exception>
#include <utility>

using namespace std;
using namespace window;

void window::GraphFrame::AdderProc() noexcept
{
	try
	{
		while (dataPipe != nullptr)
		{
			if (!dataPipe->is_empty())
			{
				while (dataPipe != nullptr && !dataPipe->is_empty())
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

		if (count == (uint64_t)4096)
		{
			trainErr->SetLightMode(true);
			testErr->SetLightMode(true);
		}

		break;

	case msg_type::count_epo:

		trainErr->Reserve(data.get_data_3());
		testErr->Reserve(data.get_data_3());

		for (uint64_t i = count; i < data.get_data_3(); ++i)
		{
			trainErr->AddData((double)i, (double)0);
			testErr->AddData((double)i, (double)0);
		}

		if (count + data.get_data_3() >= (uint64_t)4096)
		{
			trainErr->SetLightMode(true);
			testErr->SetLightMode(true);
		}

		break;

	case msg_type::count_set_1:

		trainSize = data.get_data_3();

		break;

	case msg_type::count_set_2:

		testSize = data.get_data_3();

		break;

	case msg_type::batch_mode:

		trainMode = "Batch Parallel";

		break;

	case msg_type::stoch_mode:

		trainMode = "Stochatic Seq";

		break;

	case msg_type::cross_mode:

		trainMode += " with Cross Testing";

		break;

	case msg_type::max_epo_reached:

		dataPipe = nullptr;
		trainRes = "Maximum Epoch Count reached";

		break;

	case msg_type::max_err_reached:

		dataPipe = nullptr;
		trainRes = "Maximum Test Error reached for " + to_string(data.get_data_3()) + " epoch(s)";

		break;

	case msg_type::min_err_reached:

		dataPipe = nullptr;
		trainRes = "Minimum Train Error reached for " + to_string(data.get_data_3()) + " epoch(s)";

		break;

	default:
		break;
	}

	UpdateStatusBar();
}

void window::GraphFrame::UpdateStatusBar()
{
	wxGetApp().CallAfter([this]() mutable noexcept
		{
			try
			{
				lock_guard<mutex> lock(this->adderMutex);

				this->SetStatusText(_((string("Train Mode: ") + this->trainMode).c_str()), 0);
				this->SetStatusText(_((string("Train Set: ") + to_string(this->trainSize)).c_str()), 1);
				this->SetStatusText(_((string("Test Set: ") + to_string(this->testSize)).c_str()), 2);
				this->SetStatusText(_(this->trainRes.c_str()), 3);
			}
			catch (...) {}
		});
}

window::GraphFrame::GraphFrame(list<info>&& data, pipe<info>* data_pipe, string&& name)
	: wxFrame(nullptr, wxID_ANY, _(name.c_str()), wxDefaultPosition, wxSize(800, 400),
		wxCAPTION | wxCLIP_CHILDREN | wxCLOSE_BOX | wxSYSTEM_MENU | wxMINIMIZE_BOX
		| wxMAXIMIZE_BOX | wxRESIZE_BORDER, _(name.c_str())),
	count((uint64_t)0),
	dataPipe(data_pipe),
	adderMutex(),
	adderProc(),
	axisX(new mpScaleX(_("Epoch"))),
	axisY(new mpScaleY(_("Error"))),
	infoLegend(new mpInfoLegend(wxRect(640, 10, 125, 40))),
	trainErr(new mpFXYVector(_("Train Set Error"))),
	testErr(new mpFXYVector(_("Test Set Error"))),
	plotterWnd(this, wxID_ANY, wxPoint(0, 0), wxSize(800, 400), wxSUNKEN_BORDER),
	trainMode(),
	trainRes("Processing..."),
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

	trainErr->SetPen(wxPen(wxColour((wxColourBase::ChannelType)0x00, (wxColourBase::ChannelType)0x4D,
		(wxColourBase::ChannelType)0xFF, (wxColourBase::ChannelType)255), 3, wxPenStyle::wxPENSTYLE_STIPPLE_MASK_OPAQUE));
	testErr->SetPen(wxPen(wxColour((wxColourBase::ChannelType)0xFF, (wxColourBase::ChannelType)0x53,
		(wxColourBase::ChannelType)0x00, (wxColourBase::ChannelType)255), 3, wxPenStyle::wxPENSTYLE_STIPPLE_MASK_OPAQUE));

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

	if (dataPipe != nullptr)
		adderProc = thread([this] { this->AdderProc(); });

	CreateStatusBar(4);
	UpdateStatusBar();
}

window::GraphFrame::~GraphFrame()
{
	dataPipe = nullptr;

	if (adderProc.joinable())
		adderProc.join();
}

window::NNIOFrame::NNIOFrame()
{
}

window::NNIOFrame::~NNIOFrame()
{
}

bool window::wxWndProc::OnInit()
{
	return true;
}

int window::wxWndProc::OnExit()
{
	return 0;
}

namespace window
{
	wxIMPLEMENT_APP_NO_MAIN(wxWndProc);
}