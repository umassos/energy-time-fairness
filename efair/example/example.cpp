//
// Created by tx2 on 2/20/23.
//

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "util/chfreq.h"
#include "executor/executor.h"

#define RESNET18_LIB_PATH MODEL_DIR "/resnet18/resnet18.so"
#define RESNET18_PROFILE_PATH MODEL_DIR "/resnet18/resnet18_profile.json"

int main(int argc, char **argv){

    tvm::Device dev;
    if (tvm::runtime::Registry::Get("device_api.cuda")){
        dev = {kDLCUDA, 0};
    } else {
        dev = {kDLCPU};
    }

    std::chrono::steady_clock::time_point start_t, end_t;
    efair::executor::Executor executor(RESNET18_LIB_PATH, RESNET18_PROFILE_PATH, dev);
//    efair::util::FrequencyController freq_controller;
//
//    freq_controller.set_cur_frequency_by_index(12);

    for (auto i = 0; i < 30; i++){
        executor.execute();
        executor.sync();
    }

    start_t = std::chrono::steady_clock::now();
    executor.execute();
    executor.sync();
    end_t = std::chrono::steady_clock::now();
    std::cout << "Execute latency: " << std::chrono::duration_cast<std::chrono::microseconds>(end_t-start_t).count() << "\n";

    efair::MicroSeconds time_used;
    efair::MicroJoule energy_used;
    std::string freq = "1300500000";

    start_t = std::chrono::steady_clock::now();
    executor.execute(freq, time_used, energy_used);
    executor.sync();
    end_t = std::chrono::steady_clock::now();
    std::cout << "Execute and profile latency: " << std::chrono::duration_cast<std::chrono::microseconds>(end_t-start_t).count() << "\n";

    return 0;
}