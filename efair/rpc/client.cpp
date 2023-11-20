//
// Created by Qianlin Liang on 3/24/23.
//

#include <chrono>
#include <grpc++/grpc++.h>
#include "rpc/client.h"

namespace efair {
namespace rpc {

    EFairClient::EFairClient(std::string address, std::string model_path, std::string model_profile_path,
                             std::string freq, int priority) :
                             freq(freq),
                             priority(priority){

        auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
        stub = EFairService::NewStub(channel);

        grpc::ClientContext create_entity_context;
        CreateEntityRequest create_entity_request;
        CreateEntityResponse create_entity_response;
        create_entity_request.set_priority(priority);
        grpc::Status s = stub->CreateEntity(&create_entity_context, create_entity_request, &create_entity_response);

        if (s.ok() && create_entity_response.success()) {
            std::cout << "Created entity #" << create_entity_response.eid() << std::endl;
            eid = create_entity_response.eid();
        } else
            std::cerr << "Create entity fail: " << s.error_message() << std::endl;


        grpc::ClientContext load_model_context;
        LoadModelRequest load_model_request;
        LoadModelResponse load_model_response;
        load_model_request.set_eid(eid);
        load_model_request.set_model_path(model_path);
        load_model_request.set_model_profile_path(model_profile_path);
        load_model_request.set_frequency(freq);

        s = stub->LoadModel(&load_model_context, load_model_request, &load_model_response);

        if (s.ok() && load_model_response.success()){
            mid = load_model_response.mid();
            std::cout << "Loaded model #" << mid << std::endl;
        } else {
            std::cerr << "Load model fail: " << s.error_message() << std::endl;
            std::terminate();
        }
    }

    bool EFairClient::infer() {
        grpc::ClientContext context;

        InferRequest request;
        InferResponse response;
        request.set_mid(mid);

        auto start_t = std::chrono::steady_clock::now();
        grpc::Status s = stub->Infer(&context, request, &response);
        auto end_t = std::chrono::steady_clock::now();

        if (s.ok() && response.success()){
            std::cout << "Infer finished in "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end_t-start_t).count() << " µs "
                      << "task id " << response.tid() << std::endl;
        } else {
            std::cerr << "Infer fail: " << s.error_details() << std::endl;
            return false;
        }

        return true;
    }
}   // namespace rpc
}   // namespace efair