//
// Created by ubuntu on 2/28/23.
//

#include <tvm/runtime/data_type.h>

#include "executor.h"


namespace efair {
namespace executor {

    Executor::Executor(const std::string &model_filename, tvm::Device dev) {
        // Load model
        tvm::runtime::Module module_factory = tvm::runtime::Module::LoadFromFile(model_filename);
        _module = module_factory.GetFunction("default")(dev);
        _device = dev;

        _set_input_fn = _module.GetFunction(SET_INPUT_FUNC_NAME);
        _get_output_fn = _module.GetFunction(GET_OUTPUT_FUNC_NAME);
        _execute_fn = _module.GetFunction(EXECUTE_FUNC_NAME);
        _execute_kernel_fn = _module.GetFunction(EXECUTE_KERNEL_FUNC_NAME);

        model_name = model_filename;
    }
    Executor::Executor(const std::string &model_filename, const std::string &profile_filename, tvm::Device dev) {
        // Load model
        tvm::runtime::Module module_factory = tvm::runtime::Module::LoadFromFile(model_filename);
        _module = module_factory.GetFunction("default")(dev);
        _device = dev;

        _set_input_fn = _module.GetFunction(SET_INPUT_FUNC_NAME);
        _get_output_fn = _module.GetFunction(GET_OUTPUT_FUNC_NAME);
        _execute_fn = _module.GetFunction(EXECUTE_FUNC_NAME);
        _execute_kernel_fn = _module.GetFunction(EXECUTE_KERNEL_FUNC_NAME);

        // Load profile
        auto *root = new pt::ptree;
        pt::read_json(profile_filename, *root);

        _model_profile.reset(root);
        model_name = _model_profile->get<std::string>("model_name");
    }

    Status Executor::get_input_shape(const std::string &key, tvm::runtime::ShapeTuple &ret_shape) {
        auto get_input_info_fn = _module.GetFunction("get_input_info");

        if (!get_input_info_fn.defined()) return Status::Fail;
        tvm::Map<tvm::String, tvm::ObjectRef> input_info = get_input_info_fn();

        auto input_shapes = tvm::Downcast<tvm::Map<tvm::String, tvm::ObjectRef>>(input_info["shape"]);

        ret_shape = tvm::Downcast<tvm::ShapeTuple>(input_shapes[key]);

        return Status::Succeed;
    }

    Status Executor::get_input_dtype(const std::string &key, DLDataType &ret_dtype) {
        auto get_input_info_fn = _module.GetFunction("get_input_info");

        if (!get_input_info_fn.defined()) return Status::Fail;
        tvm::Map<tvm::String, tvm::ObjectRef> input_info = get_input_info_fn();

        auto input_shapes = tvm::Downcast<tvm::Map<tvm::String, tvm::ObjectRef>>(input_info["dtype"]);

        ret_dtype = tvm::runtime::String2DLDataType(tvm::Downcast<tvm::String>(input_shapes[key]));
        return Status::Succeed;
    }

    Status Executor::get_gpu_power(std::string freq, efair::MilliWatt &ret_gpu_power) {
        std::string power_path = "gpu_power." + freq;
        ret_gpu_power = _model_profile->get<MilliWatt>(power_path);
        return Status::Succeed;
    }

    Status Executor::get_max_gpu_power(MilliWatt &ret_power) {
        float cur_power = 0, max_power = 0;

        for (const auto & [freq, power] : _model_profile->get_child("gpu_power")){
            cur_power = std::stof(power.data());
            if (cur_power > max_power)
                max_power = cur_power;
        }

        ret_power = static_cast<MilliWatt>(max_power);
        return Status::Succeed;
    }

    Status Executor::set_input(const std::string &key, const void *input_data, size_t size) {
        tvm::ShapeTuple shape;
        DLDataType dtype;
        tvm::Device cpu{kDLCPU, 0};

        RETURN_STATUS(get_input_shape(key, shape));
        RETURN_STATUS(get_input_dtype(key, dtype));

        size_t data_size = 1;
        for (auto& s : shape){
            data_size *= s;
        }
        data_size = data_size * (dtype.bits / 8);

        if (size > data_size){
            LOG(ERROR) << "Copy input bytes fail, expect input size " << data_size << " but get " << size;
            return Status::Fail;
        }

        tvm::runtime::NDArray input_ndarray = tvm::runtime::NDArray::Empty(shape, dtype, cpu);
        input_ndarray.CopyFromBytes(input_data, size);

        RETURN_STATUS(set_input(key, input_ndarray));
        return Status::Succeed;
    }

    Status Executor::set_input(const std::string &key, const tvm::runtime::NDArray &input_data) {
        if (!_set_input_fn.defined()){
            LOG(ERROR) << "PackedFunction set_input is not defined.";
            return Status::Fail;
        }
        _set_input_fn(key, input_data);
        return Status::Succeed;
    }


    Status Executor::get_output(size_t idx, tvm::runtime::NDArray &out) {
        int num_outputs = _module.GetFunction("get_num_outputs")();
        if (idx >= num_outputs){
            LOG(ERROR) << "Get output out of range, totally " << num_outputs << " but getting index " << idx;
            return Status::Fail;
        }

        _get_output_fn(idx, out);
        return Status::Succeed;
    }

    void Executor::execute() {
        _execute_fn();
    }

    void Executor::execute(const std::string &freq, efair::MicroSeconds &time_used, efair::MicroJoule &energy_used) {
        execute();
        std::string time_json_path = "exec_time." + freq;
        std::string energy_json_path = "energy." + freq;


        time_used = _model_profile->get<efair::MicroSeconds>(time_json_path);
        energy_used = _model_profile->get<efair::MicroJoule>(energy_json_path);
    }

    void Executor::execute_kernel(const size_t &idx) {
        _execute_kernel_fn(idx);
    }

    void Executor::execute_kernel(const size_t &idx, const std::string &freq, efair::MicroSeconds &time_used,
                                  efair::MicroJoule &energy_used) {
        execute_kernel(idx);

        std::string kernel_name;
        MilliWatt cur_gpu_power;
        get_kernel_name(idx, kernel_name);

        std::string time_json_path = "kernel_profile." + kernel_name + ".exec_time." + freq;
//        std::string energy_json_path = "kernel_profile." + kernel_name + ".energy." + freq;

        time_used = _model_profile->get<efair::MicroSeconds>(time_json_path);
        get_gpu_power(freq, cur_gpu_power);
        energy_used = cur_gpu_power * time_used * 1e-3;
//        energy_used = _model_profile->get<efair::MicroJoule>(energy_json_path);
    }

    Status Executor::get_num_kernels(size_t &n) {
        tvm::runtime::PackedFunc get_num_kernels_fn = _module.GetFunction(GET_NUM_KERNELS_FUNC_NAME);

        if (!get_num_kernels_fn.defined()) return Status::Fail;

        int k = get_num_kernels_fn();
        n = (size_t) k;
        return Status::Succeed;
    }

    Status Executor::get_kernel_name(size_t idx, std::string &kernel_name) {
        tvm::runtime::PackedFunc get_kernel_name_fn = _module.GetFunction(GET_KERNEL_NAME_FUNC_NAME);

        if (!get_kernel_name_fn.defined()) return Status::Fail;

        kernel_name = static_cast<tvm::String>(get_kernel_name_fn(idx));
        return Status::Succeed;
    }

    void Executor::sync() {
        TVMSynchronize(_device.device_type, _device.device_id, _stream);
    }

}   // namespace executor
}   // namespace efair