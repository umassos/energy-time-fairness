//
// Created by tx2 on 3/1/23.
//

#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <fstream>

#include "util/chfreq.h"
#include "executor/executor.h"


template<typename T>
T mean(const std::vector<T>& vec){
    if (vec.empty()) return 0;

    return std::accumulate(vec.begin(), vec.end(), 0) / vec.size();
}

typedef struct {
        size_t exec_time;
        size_t power;
        std::string frequency;
        std::string kernel_name;
    } ProfileRecord;

int main(int argc, char** argv){
    if (argc < 2) {
        LOG(ERROR) << "Target file not specified, exiting.";
        exit(1);
    }

    efair::util::FrequencyController freq_controller;
    std::vector<std::string> available_frequencies;
    ASSERT_STATUS(freq_controller.get_available_frequencies(available_frequencies));

    tvm::Device dev{kDLCUDA, 0};
    efair::executor::Executor executor(argv[1], dev);

    size_t num_kernels;
    ASSERT_STATUS(executor.get_num_kernels(num_kernels));

    std::chrono::steady_clock::time_point start_t, kernel_start_t;

    size_t time_limit = 5000000;  // 5 seconds
    std::string kernel_name, gpu_frequency;
    size_t power;
    std::vector<ProfileRecord> records;

    std::vector<size_t> exec_time, gpu_power;


    for (const auto& gpu_frequency : available_frequencies) {
        LOG(INFO) << "Changing GPU frequency to " << gpu_frequency;
        freq_controller.set_cur_frequency(gpu_frequency);

        // Profile kernels
        for (auto i = 0; i < num_kernels; i++) {
            exec_time.clear();
            gpu_power.clear();
            start_t = std::chrono::steady_clock::now();

            while (std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start_t).count() < time_limit) {
                kernel_start_t = std::chrono::steady_clock::now();
                executor.execute_kernel(i);
                executor.sync();
                exec_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - kernel_start_t).count());
                ASSERT_STATUS(freq_controller.get_gpu_power(power));
                gpu_power.push_back(power);
            }

            size_t avg_exec_time = mean(exec_time);
            size_t avg_gpu_power = mean(gpu_power);
            ASSERT_STATUS(executor.get_kernel_name(i, kernel_name));

            LOG(INFO) << "Kernel " << kernel_name << " execution time: " << avg_exec_time << " μs, power: "
                      << avg_gpu_power << " mW";
            records.push_back(ProfileRecord{
                    avg_exec_time,
                    avg_gpu_power,
                    gpu_frequency,
                    kernel_name
            });
        }

        // Profile model end to end
        exec_time.clear();
        gpu_power.clear();
        start_t = std::chrono::steady_clock::now();

        while (std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_t).count() < time_limit) {
            kernel_start_t = std::chrono::steady_clock::now();
            executor.execute();
            executor.sync();
            exec_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - kernel_start_t).count());
            ASSERT_STATUS(freq_controller.get_gpu_power(power));
            gpu_power.push_back(power);
        }
        size_t avg_exec_time = mean(exec_time);
        size_t avg_gpu_power = mean(gpu_power);
        LOG(INFO) << "End to end" << " execution time: " << avg_exec_time << " μs, power: " << avg_gpu_power << " mW";
        records.push_back(ProfileRecord{
                avg_exec_time,
                avg_gpu_power,
                gpu_frequency,
                "ALL"
        });
    }


    // Write records to file
    std::string out_path = std::string(argv[1]) + ".profile.csv";
    std::ofstream out_file(out_path);

    if (!out_file.is_open()) {
        LOG(ERROR) << "Cannot write to file " << out_path;
        exit(1);
    }

    out_file << "frequency,exec_time,gpu_power,kernel_name\n";
    for (const auto& record : records){
        out_file << record.frequency << "," << record.exec_time << "," << record.power << "," << record.kernel_name << "\n";
    }

    return 0;
}