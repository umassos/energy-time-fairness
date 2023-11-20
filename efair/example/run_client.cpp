//
// Created by Qianlin Liang on 3/24/23.
//

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <csignal>
#include <chrono>

#include "rpc/client.h"

bool shutdown_ = false;

void interrupt_handler(int signum) {
    std::cout << "Received interrupt signal " << signum << std::endl;
    shutdown_ = true;
};

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "Expected arguments [model_path] [model_profile_path] [frequency_idx] [priority] [num_threads] "
                  << "with optional [delay start time] and [duration]"
                  << std::endl;
        std::exit(1);
    }

    std::vector<std::string> avai_frequencies = {"114750000", "216750000", "318750000", "420750000", "522750000",
                                                 "624750000", "726750000", "854250000", "930750000", "1032750000",
                                                 "1122000000", "1236750000", "1300500000"};


    std::string model_path = argv[1];
    std::string model_profile_path = argv[2];
    std::string frequency = avai_frequencies[std::atoi(argv[3])];
    int priority = std::atoi(argv[4]);
    int num_threads = std::atoi(argv[5]);
    int delay_start_time_second = 0, duration_second = 0;

    if (argc > 6)
        delay_start_time_second = std::atoi(argv[6]);
    if (argc > 7)
        duration_second = std::atoi(argv[7]);

    if (delay_start_time_second > 0)
        std::this_thread::sleep_for(std::chrono::seconds(delay_start_time_second));

    std::signal(SIGINT, interrupt_handler);

    efair::rpc::EFairClient client(SERVER_ADDRESS, model_path, model_profile_path, frequency, priority);

    std::vector<std::thread> thread_pool;

    for (auto i = 0; i < num_threads; i++){
        std::thread t([&](){
            while (!shutdown_) {
                if (!client.infer())
                    shutdown_ = true;
            }
        });

        thread_pool.push_back(std::move(t));
    }

    if (duration_second > 0){
        std::this_thread::sleep_for(std::chrono::seconds(duration_second));
        std::cout << "Timeout, quiting..." << std::endl;
        shutdown_ = true;
    }

    for (auto i = 0; i < num_threads; i++){
        thread_pool[i].join();
    }

    return 0;
}

