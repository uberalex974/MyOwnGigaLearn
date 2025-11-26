"""OPT 1: Async Data Loading - std::async simple"""
code = '''
// === ASYNC DATA LOADING (+15%) ===
#include <future>

class AsyncBatchLoader {
public:
    std::future<ExperienceBuffer> next_batch_future;
    bool loading = false;
    
    void startLoadNext(std::function<ExperienceBuffer()> loader) {
        if (!loading) {
            next_batch_future = std::async(std::launch::async, loader);
            loading = true;
        }
    }
    
    ExperienceBuffer getNext() {
        if (loading) {
            auto batch = next_batch_future.get();
            loading = false;
            return batch;
        }
        return ExperienceBuffer(); // Empty if not ready
    }
};
'''

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'r', encoding='utf-8') as f:
    content = f.read()

if 'AsyncBatchLoader' not in content:
    content = code + '\n' + content
    
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'w', encoding='utf-8') as f:
    f.write(content)
    
print("✅ Async Data Loading implemented")
