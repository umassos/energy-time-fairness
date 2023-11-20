//
// Created by ubuntu on 2/28/23.
//

#ifndef EFAIR_EXECUTOR_H
#define EFAIR_EXECUTOR_H

#include <string>
#include <vector>
#include <chrono>

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/device_api.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "util/common.h"

#define SET_INPUT_FUNC_NAME "set_input"
#define GET_OUTPUT_FUNC_NAME "get_output"
#define EXECUTE_FUNC_NAME "execute"
#define EXECUTE_KERNEL_FUNC_NAME "execute_kernel"
#define GET_NUM_KERNELS_FUNC_NAME "get_num_kernels"
#define GET_KERNEL_NAME_FUNC_NAME "get_kernel_name"

namespace pt = boost::property_tree;

namespace efair {
namespace executor {
    class Executor {
    public:
        std::string model_name;

        Executor() = delete;
        Executor(const std::string &model_filename, tvm::Device dev);
        Executor(const std::string &model_filename, const std::string &profile_filename, tvm::Device dev);
        ~Executor() = default;

        Status get_input_shape(const std::string& key, tvm::runtime::ShapeTuple& ret_shape);
        Status get_input_dtype(const std::string& key, DLDataType& ret_dtype);
        Status get_max_gpu_power(MilliWatt &ret_power);
        Status get_gpu_power(std::string freq, MilliWatt &ret_gpu_power);

        Status set_input(const std::string& key, const tvm::runtime::NDArray& input_data);
        Status set_input(const std::string& key, const void* input_data, size_t size);

        Status get_output(size_t idx, tvm::runtime::NDArray& out);

        template<typename T>
        Status get_output(size_t idx, std::vector<T>& out) {
            DLDevice cpu{kDLCPU};

            tvm::runtime::NDArray out_array = static_cast<tvm::runtime::NDArray>(_get_output_fn(idx)).CopyTo(cpu);
            if (out_array->dtype.bits / 8 != sizeof(T)){
                LOG(ERROR) << "Types not compatible";
                return Status::Fail;
            }

            size_t output_size = 1;
            for (auto i = 0; i < out_array->ndim; ++i){
                output_size *= out_array->shape[i];
            }

            out.clear();
            for (auto i = 0; i < output_size; ++i){
                out.push_back(static_cast<T*>(out_array->data)[i]);
            }

            return Status::Succeed;
        }

        void execute(void);
        void execute(const std::string &freq, efair::MicroSeconds &time_used, efair::MicroJoule &energy_used);
        void execute_kernel(const size_t &idx);
        void execute_kernel(const size_t &idx, const std::string &freq, efair::MicroSeconds& time_used,
                            efair::MicroJoule &energy_used);
        Status get_num_kernels(size_t &n);
        Status get_kernel_name(size_t idx, std::string &kernel_name);

        void sync(void);

    private:

        tvm::runtime::Module _module;
        tvm::Device _device{};
        TVMStreamHandle _stream = nullptr;

        tvm::runtime::PackedFunc _set_input_fn;
        tvm::runtime::PackedFunc _get_output_fn;
        tvm::runtime::PackedFunc _execute_fn;
        tvm::runtime::PackedFunc _execute_kernel_fn;

        std::unique_ptr<pt::ptree> _model_profile;

    };

}   // namespace executor
}   // namespace efair

#endif //EFAIR_EXECUTOR_H


