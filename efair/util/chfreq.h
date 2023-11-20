//
// Created by tx2 on 2/20/23.
//

#ifndef EFAIR_CHFREQ_H
#define EFAIR_CHFREQ_H

#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <thread>
#include <condition_variable>
#include "util/common.h"

#define MIN_FREQUENCY_FILE "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/min_freq"
#define CUR_FREQUENCY_FILE "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/cur_freq"
#define MAX_FREQUENCY_FILE "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/max_freq"
#define AVAILABLE_FREQUENCY_FILE "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/available_frequencies"
#define GPU_POWER_FILE "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input"

namespace efair {
namespace util {

    class FrequencyController {
    public:
        FrequencyController();
        ~FrequencyController();

        Status get_frequency(std::string& freq);
        Status set_cur_frequency(const std::string &freq);
        Status set_cur_frequency_by_index(const size_t &idx);
        Status get_available_frequencies(std::vector<std::string> &ret_freq);
        Status get_gpu_power(size_t& gpu_power);

        void shutdown();

    private:

        Status write_frequency(const std::string& filename, const std::string& freq);
        Status read_frequency_from_file(const std::string& filename);
        Status set_cur_frequency_internal(const std::string &freq);

        void loop_body();

        bool shutdown_requested;
        std::mutex freq_lock;
        std::condition_variable cv;
        std::unique_ptr<std::thread> worker_thread;

        std::string target_frequency;
        std::string cur_frequency;
        std::unordered_map<std::string, size_t> freq2idx;
        std::unordered_map<size_t, std::string> idx2freq;
    };

} // namespace util
} // namespace efair

#endif //EFAIR_CHFREQ_H