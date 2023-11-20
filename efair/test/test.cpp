//
// Created by ubuntu on 3/1/23.
//

#include <fstream>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "util/common.h"
#include "executor/executor.h"
#include "scheduler/scheduler.h"

#define ASSERT_SUCC(expr) ASSERT_TRUE(expr == efair::Status::Succeed)

#define RESNET18_LIB_PATH MODEL_DIR "/resnet18/resnet18.so"
#define RESNET18_PROFILE_PATH MODEL_DIR "/resnet18/resnet18_profile.json"
#define RESNET50_LIB_PATH MODEL_DIR "/resnet50/resnet50.so"
#define RESNET50_PROFILE_PATH MODEL_DIR "/resnet50/resnet50_profile.json"
#define CHIHUAHUA_IMAGE_FILEPATH SAMPLE_DIR "/image_chihuahua.bytes"


class ExecutorTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (tvm::runtime::Registry::Get("device_api.cuda")){
            dev = {kDLCUDA, 0};
        } else {
            dev = {kDLCPU};
        }

        resnet18_executor = std::make_shared<efair::executor::Executor>(RESNET18_LIB_PATH, RESNET18_PROFILE_PATH, dev);
        resnet50_executor = std::make_shared<efair::executor::Executor>(RESNET50_LIB_PATH, RESNET50_PROFILE_PATH, dev);

        std::ifstream input_file(CHIHUAHUA_IMAGE_FILEPATH, std::ios::binary);
        input_file.seekg(0, std::ios::end);
        size_t file_length = input_file.tellg();
        input_file.seekg(0, std::ios::beg);

        input_buffer = new char[file_length];
        input_length = file_length;
        input_file.read(input_buffer, file_length);
        input_file.close();
    }

    tvm::Device dev{};
    std::shared_ptr<efair::executor::Executor> resnet18_executor;
    std::shared_ptr<efair::executor::Executor> resnet50_executor;
    char *input_buffer{};
    size_t input_length{};
};

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (tvm::runtime::Registry::Get("device_api.cuda")){
            dev = {kDLCUDA, 0};
        } else {
            dev = {kDLCPU};
        }

        freq = "1300500000";
        scheduler = std::make_shared<efair::scheduler::EFairScheduler>(40000, 1.0, dev);
    }

    tvm::Device dev;
    std::shared_ptr<efair::scheduler::EFairScheduler> scheduler;
    std::string freq;

};

TEST_F(SchedulerTest, schedulerOperations){
    // Create entity
    efair::EntityID eid;
    ASSERT_SUCC(scheduler->create_entity(0, eid));
    ASSERT_EQ(eid, 0);

    // Load model
    efair::ModelID mid;
    ASSERT_SUCC(scheduler->load_model(RESNET18_LIB_PATH, RESNET18_PROFILE_PATH, eid, freq, mid));
    ASSERT_EQ(mid, 0);

    // New task
    efair::TaskID tid;
    ASSERT_SUCC(scheduler->new_task(mid, tid));
    ASSERT_EQ(tid, 0);

    // run
    ASSERT_SUCC(scheduler->run());

    // wait task
    ASSERT_SUCC(scheduler->wait_task(tid));

    // shutdown
    ASSERT_SUCC(scheduler->shutdown());

}


TEST_F(ExecutorTest, getNumKernels){
    size_t num_kernels;
    ASSERT_SUCC(resnet18_executor->get_num_kernels(num_kernels));
    ASSERT_EQ(num_kernels, 24);
}

TEST_F(ExecutorTest, getInputShape){
    tvm::runtime::ShapeTuple ret_shape;
    tvm::runtime::ShapeTuple expect_shape{1, 3, 224, 224};

    // Resnet18
    ASSERT_SUCC(resnet18_executor->get_input_shape("input", ret_shape));
    ASSERT_EQ(expect_shape.size(), ret_shape.size());

    for (auto i = 0; i < expect_shape.size(); ++i){
        ASSERT_EQ(expect_shape[i], ret_shape[i]);
    }

    // Resnet50
    ASSERT_SUCC(resnet50_executor->get_input_shape("input", ret_shape));
    ASSERT_EQ(expect_shape.size(), ret_shape.size());

    for (auto i = 0; i < expect_shape.size(); ++i){
        ASSERT_EQ(expect_shape[i], ret_shape[i]);
    }
}

TEST_F(ExecutorTest, getDType){
    DLDataType ret_dtype;
    DLDataType expect_dtype{kDLFloat, 32, 1};

    // Resnet18
    ASSERT_SUCC(resnet18_executor->get_input_dtype("input", ret_dtype));
    ASSERT_EQ(ret_dtype.code, expect_dtype.code);

    // Resnet50
    ASSERT_SUCC(resnet50_executor->get_input_dtype("input", ret_dtype));
    ASSERT_EQ(ret_dtype.code, expect_dtype.code);
}

TEST_F(ExecutorTest, executeResnet18){
    ASSERT_SUCC(resnet18_executor->set_input("input", input_buffer, input_length));

    efair::MicroSeconds time_used;
    efair::MicroJoule energy_used;
    std::string freq = "1300500000";
    resnet18_executor->execute(freq, time_used, energy_used);

    ASSERT_EQ(time_used, 21406);
    ASSERT_EQ(energy_used, 79973);

    std::vector<float> ret_vector;
    ASSERT_SUCC(resnet18_executor->get_output(0, ret_vector));
    ASSERT_EQ(ret_vector.size(), 1000);

    float max_val = -1000;
    size_t max_idx = 0;

    for (size_t i = 0; i < ret_vector.size(); i++){
        if (ret_vector[i] > max_val){
            max_val = ret_vector[i];
            max_idx = i;
        }
    }

    ASSERT_EQ(max_idx, 151);
}

TEST_F(ExecutorTest, executeResnet50){
    ASSERT_SUCC(resnet50_executor->set_input("input", input_buffer, input_length));

    efair::MicroSeconds time_used;
    efair::MicroJoule energy_used;
    std::string freq = "1300500000";
    resnet50_executor->execute(freq, time_used, energy_used);

    ASSERT_EQ(time_used, 57005);
    ASSERT_EQ(energy_used, 242902);

    std::vector<float> ret_vector;
    ASSERT_SUCC(resnet50_executor->get_output(0, ret_vector));
    ASSERT_EQ(ret_vector.size(), 1000);

    float max_val = -1000;
    size_t max_idx = 0;

    for (size_t i = 0; i < ret_vector.size(); i++){
        if (ret_vector[i] > max_val){
            max_val = ret_vector[i];
            max_idx = i;
        }
    }

    ASSERT_EQ(max_idx, 151);
}

TEST_F(ExecutorTest, getKernelName){
    std::string kernel_name;
    ASSERT_SUCC(resnet18_executor->get_kernel_name(0, kernel_name));
    ASSERT_SUCC(resnet50_executor->get_kernel_name(0, kernel_name));
    ASSERT_SUCC(resnet18_executor->get_kernel_name(5, kernel_name));
    ASSERT_SUCC(resnet50_executor->get_kernel_name(5, kernel_name));
}

