import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def optimize_metric_sender():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Util\MetricSender.cpp"
    content = read_file(path)
    
    # Make Send async
    # We need to include <thread> if not present
    if "#include <thread>" not in content:
        content = "#include <thread>\n" + content
        
    if "std::thread" not in content:
        print("Optimizing MetricSender: Making Send asynchronous")
        # Replace the body of Send
        # Original:
        # void GGL::MetricSender::Send(const Report& report) {
        # 	py::dict reportDict = {};
        # 	for (auto& pair : report.data)
        # 		reportDict[pair.first.c_str()] = pair.second;
        # 	try {
        # 		pyMod.attr("add_metrics")(reportDict);
        # 	} catch (std::exception& e) {
        # 		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
        # 	}
        # }
        
        # New:
        # void GGL::MetricSender::Send(const Report& report) {
        #     // Copy report data to avoid lifetime issues
        #     Report reportCopy = report;
        #     std::thread([this, reportCopy]() {
        #         // Acquire GIL for Python operations
        #         py::gil_scoped_acquire acquire;
        #         py::dict reportDict = {};
        #         for (auto& pair : reportCopy.data)
        #             reportDict[pair.first.c_str()] = pair.second;
        #         try {
        #             pyMod.attr("add_metrics")(reportDict);
        #         } catch (std::exception& e) {
        #             // Log error but don't crash main thread
        #             RG_LOG("MetricSender Error: " << e.what());
        #         }
        #     }).detach();
        # }
        
        # We need to be careful about GIL. `pybind11::embed` usually runs in the main thread.
        # If we spawn a thread, we MUST acquire GIL.
        # Also, `pyMod` access needs to be safe.
        # Actually, `pybind11` documentation says: "If you want to call Python functions from other C++ threads, you must hold the GIL."
        # So `py::gil_scoped_acquire` is necessary.
        
        # However, `MetricSender.cpp` doesn't seem to include `pybind11/embed.h` directly, but `MetricSender.h` might.
        # `MetricSender.cpp` has `namespace py = pybind11;`.
        
        # Let's construct the replacement.
        
        new_send = """void GGL::MetricSender::Send(const Report& report) {
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
}"""
        
        # Regex to replace the function body
        pattern = r"void GGL::MetricSender::Send\(const Report& report\) \{.*?\}"
        content = re.sub(pattern, new_send, content, flags=re.DOTALL)
        
    write_file(path, content)

def optimize_learner_cpp():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"
    content = read_file(path)
    
    # Parallelize Obs Norm Loop
    # Look for the loop
    # for (int i = 0; i < envSet->state.numPlayers; i++) {
    #     for (int j = 0; j < obsSize; j++) {
    #         float& obsVal = envSet->state.obs.At(i, j);
    #         obsVal = (obsVal - mean[j]) / std[j];
    #     }
    # }
    
    if "#pragma omp parallel for" not in content:
        # We need to find the specific loop for obs normalization.
        # It's inside `if (!render && obsStat)` block.
        
        # Let's search for the inner loop structure or the `At(i, j)` call.
        
        # We can replace the whole block if we can identify it.
        # "float& obsVal = envSet->state.obs.At(i, j);"
        
        # Replacement with OpenMP and direct pointer access
        # float* data = envSet->state.obs.data.data();
        # #pragma omp parallel for
        # for (int i = 0; i < envSet->state.numPlayers; i++) {
        #     for (int j = 0; j < obsSize; j++) {
        #         float& obsVal = data[i * obsSize + j];
        #         obsVal = (obsVal - mean[j]) / std[j];
        #     }
        # }
        
        if "float& obsVal = envSet->state.obs.At(i, j);" in content:
            print("Optimizing Learner.cpp: Parallelizing Obs Norm")
            
            # We need to capture `mean`, `std`, `obsSize` in the lambda/block.
            # OpenMP handles this.
            
            # We'll replace the loop.
            original_loop = """for (int i = 0; i < envSet->state.numPlayers; i++) {
								for (int j = 0; j < obsSize; j++) {
									float& obsVal = envSet->state.obs.At(i, j);
									obsVal = (obsVal - mean[j]) / std[j];
								}
							}"""
            
            # Note: The original code might have different indentation.
            # Let's use regex to be safe.
            pattern = r"for \(int i = 0; i < envSet->state.numPlayers; i\+\+\) \{\s*for \(int j = 0; j < obsSize; j\+\+\) \{\s*float& obsVal = envSet->state.obs.At\(i, j\);\s*obsVal = \(obsVal - mean\[j\]\) / std\[j\];\s*\}\s*\}"
            
            new_loop = """// Optimization: Parallelize Obs Norm with OpenMP and direct access
							float* rawData = envSet->state.obs.data.data();
							#pragma omp parallel for
							for (int i = 0; i < envSet->state.numPlayers; i++) {
								for (int j = 0; j < obsSize; j++) {
									// Direct pointer access is faster than At()
									float& obsVal = rawData[i * obsSize + j];
									obsVal = (obsVal - mean[j]) / std[j];
								}
							}"""
            
            content = re.sub(pattern, new_loop, content, flags=re.DOTALL)
            
    write_file(path, content)

def main():
    print("Applying Advanced Bottleneck Optimizations...")
    optimize_metric_sender()
    optimize_learner_cpp()
    print("Done.")

if __name__ == "__main__":
    main()
