#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <thread>

#include "info.h"
#include "pipe.h"
#include "wx/wx.h"
#include "mathplot.h"

namespace window
{
	class GraphFrame : public wxFrame
	{
	private:
		uint64_t count;
		pipe<info>* dataPipe;
		std::mutex adderMutex;
		std::thread adderProc;

		mpScaleX* axisX;
		mpScaleY* axisY;
		mpInfoLegend* infoLegend;
		mpFXYVector* trainErr;
		mpFXYVector* testErr;
		mpWindow plotterWnd;

		std::string trainMode;
		std::string trainRes;
		uint64_t trainSize;
		uint64_t testSize;

		void AdderProc() noexcept;
		void ProcInfo(const info& data);
		void UpdateStatusBar();

	public:
		GraphFrame() = delete;
		GraphFrame(std::list<info>&& data, pipe<info>* data_pipe, std::string&& name);
		GraphFrame(const GraphFrame& o) = delete;
		GraphFrame(GraphFrame&& o) = delete;
		~GraphFrame();
	};

	class NNIOFrame : public wxFrame
	{
	private:
	public:
		//NNIOFrame()=delete;
		NNIOFrame();
		NNIOFrame(const NNIOFrame& o) = delete;
		NNIOFrame(NNIOFrame&& o) = delete;
		~NNIOFrame();
	};

	class wxWndProc : public wxApp
	{
	public:
		bool OnInit() override;
		int OnExit() override;
	};

	wxDECLARE_APP(wxWndProc);
}