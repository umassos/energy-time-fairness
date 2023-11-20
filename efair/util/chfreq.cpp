//
// Created by tx2 on 2/20/23.
//

#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <unistd.h>

#include "util/chfreq.h"
#include "util/common.h"

namespace efair {
    namespace util {

        Status FrequencyController::read_frequency_from_file(const std::string& filename) {
            std::ifstream cur_freq_file(filename);

            if (cur_freq_file.is_open()){
                std::getline(cur_freq_file, cur_frequency);

                cur_freq_file.close();
                return Status::Succeed;
            } else {
                return Status::Fail;
            }
        }

        Status FrequencyController::get_available_frequencies(std::vector<std::string> &ret_freq) {
            if (!ret_freq.empty()){
                return Status::Fail;  // vector already has content
            }

            std::ifstream avai_freq_file (AVAILABLE_FREQUENCY_FILE);

            if (avai_freq_file.is_open()){
                for (std::string line; std::getline(avai_freq_file, line, ' '); ){
                    line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
                    ret_freq.push_back(line);
                }

                avai_freq_file.close();
                return Status::Succeed;
            } else {
                return Status::Fail;
            }
        }


        Status FrequencyController::write_frequency(const std::string& filename, const std::string &freq) {
            std::ofstream freq_file(filename);

            if (freq_file.is_open()){
                freq_file << freq;

                freq_file.close();
                return freq_file.fail()? Status::Fail : Status::Succeed;
            }

            return Status::Fail;
        }

        Status FrequencyController::get_frequency(std::string& freq) {
            freq = target_frequency;
            return Status::Succeed;
        }

        Status FrequencyController::set_cur_frequency(const std::string &freq) {
            std::unique_lock<std::mutex> lock(freq_lock);
            target_frequency = freq;
            cv.notify_one();

            return Status::Succeed;
        }

        Status FrequencyController::set_cur_frequency_internal(const std::string &freq) {
            if (!freq2idx.count(freq)){
                return Status::Fail;
            }

            int target_freq_num = std::stoi(freq);
            int cur_freq_num = std::stoi(cur_frequency);

            Status s1 = Status::Succeed, s2 = Status::Succeed;
            if (cur_freq_num > target_freq_num){
                RETURN_STATUS(write_frequency(MIN_FREQUENCY_FILE, freq));
                RETURN_STATUS(write_frequency(MAX_FREQUENCY_FILE, freq));
            } else if (cur_freq_num < target_freq_num) {
                RETURN_STATUS(write_frequency(MAX_FREQUENCY_FILE, freq));
                RETURN_STATUS(write_frequency(MIN_FREQUENCY_FILE, freq));
            }

            RETURN_STATUS(read_frequency_from_file(CUR_FREQUENCY_FILE));
            return cur_frequency == freq ? Status::Succeed : Status::Fail;
        }

        Status FrequencyController::set_cur_frequency_by_index(const size_t &idx) {
            RETURN_STATUS(set_cur_frequency(idx2freq[idx]));
            return Status::Succeed;
        }

        Status FrequencyController::get_gpu_power(size_t &gpu_power) {
            std::ifstream gpu_power_file(GPU_POWER_FILE);

            if (gpu_power_file.is_open()){
                std::string line;
                std::getline(gpu_power_file, line);

                gpu_power = std::stoul(line);

                gpu_power_file.close();
                return Status::Succeed;
            }

            return Status::Fail;
        }

        void FrequencyController::loop_body() {
            std::unique_lock<std::mutex> lock(freq_lock);
            cv.wait(lock, [&]{
                return target_frequency != cur_frequency || shutdown_requested;
            });
            ASSERT_STATUS(set_cur_frequency_internal(target_frequency));
        }

        void FrequencyController::shutdown() {
            {
                std::unique_lock<std::mutex> lock(freq_lock);
                shutdown_requested = true;
                cv.notify_one();
            }

            worker_thread->join();
            LOG(INFO) << "Frequency controller is shutdown.";
        }

        FrequencyController::FrequencyController() {
            if (getuid()) {
                throw std::runtime_error("Need root to change GPU frequency, exiting.");
            }

            ASSERT_STATUS(read_frequency_from_file(CUR_FREQUENCY_FILE));
            target_frequency = cur_frequency;
            shutdown_requested = false;

            std::vector<std::string> freq_vec;
            ASSERT_STATUS(get_available_frequencies(freq_vec));

            for (auto i = 0; i < freq_vec.size(); i++){
                idx2freq[i] = freq_vec[i];
                freq2idx[freq_vec[i]] = i;
            }

            worker_thread = std::make_unique<std::thread>([this]{
                while (!this->shutdown_requested){
                    this->loop_body();
                }
            });
        }

        FrequencyController::~FrequencyController() = default;
}
}
