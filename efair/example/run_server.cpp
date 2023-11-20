//
// Created by Qianlin Liang on 3/24/23.
//

#include <iostream>
#include <csignal>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <filesystem>
#include "rpc/server.h"

efair::scheduler::EFairScheduler *scheduler = nullptr;
efair::rpc::EFairServer *server = nullptr;
bool shutdown_requested = false;
std::mutex lk;
std::condition_variable cv;


void interrupt_handler(int signum){
    std::unique_lock<std::mutex> lock(lk);
    shutdown_requested = true;
    cv.notify_all();
}

void shutdown_server(){
    std::unique_lock<std::mutex> lock(lk);
    cv.wait(lock, []{
        return shutdown_requested;
    });

    server->shutdown();
    scheduler->summary_task_by_model();

    auto run_num = 0;
    std::filesystem::path result_folder{"./results"};

    if (!std::filesystem::exists(result_folder))
        std::filesystem::create_directories(result_folder);

    auto folder_name = "run" + std::to_string(run_num);
    while (std::filesystem::exists(result_folder/folder_name)){
        run_num++;
        folder_name = "run" + std::to_string(run_num);
    }

    std::filesystem::create_directories(result_folder/folder_name);
    scheduler->export_task_data(result_folder/folder_name/"tasks.csv");
}

int main(int argc, char **argv){
    if (argc < 4) {
        std::cerr << "Need as least 3 arguments to run server: [quantum_size] [phi] [device] " << std::endl;
        std::exit(1);
    }

    std::signal(SIGINT, interrupt_handler);

    auto quantum_size = static_cast<efair::MicroSeconds>(std::atoi(argv[1]));
    auto phi = std::atof(argv[2]);

    tvm::Device dev;
    if (std::strcmp(argv[3], "gpu") == 0){
        std::cout << "Using GPU." << std::endl;
        dev = {kDLCUDA, 0};
    } else {
        std::cout << "Using CPU" << std::endl;
        dev = {kDLCPU};
    }

    scheduler = new efair::scheduler::EFairScheduler(quantum_size, phi, dev);
    server = new efair::rpc::EFairServer(SERVER_ADDRESS, scheduler);

    std::thread t(shutdown_server);

    server->run();

    t.join();
    return 0;
}