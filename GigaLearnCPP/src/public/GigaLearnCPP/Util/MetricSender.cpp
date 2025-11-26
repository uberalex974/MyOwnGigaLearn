#include <thread>
#include "MetricSender.h"

#include "Timer.h"

namespace py = pybind11;
using namespace GGL;

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	try {
		pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
	}

	try {
		auto returedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initalized.");
}

void GGL::MetricSender::Send(const Report& report) {
	// Optimization: Async sending to prevent blocking training loop
	// Copy report data to avoid lifetime issues in detached thread
	Report reportCopy = report;
	
	std::thread([this, reportCopy]() {
		// Acquire GIL for Python operations
		py::gil_scoped_acquire acquire;
		
		py::dict reportDict = {};
		for (auto& pair : reportCopy.data)
			reportDict[pair.first.c_str()] = pair.second;

		try {
			pyMod.attr("add_metrics")(reportDict);
		} catch (std::exception& e) {
			RG_LOG("MetricSender Error: " << e.what());
		}
	}).detach();
}

GGL::MetricSender::~MetricSender() {
	
}