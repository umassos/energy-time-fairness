//
// Created by Qianlin Liang on 3/23/23.
//

#include "rpc/server.h"

namespace efair {
namespace rpc {

    EFairServer::EFairServer(std::string address, efair::scheduler::EFairScheduler *scheduler_ptr) :
    address(address)
    {
        scheduler.reset(scheduler_ptr);
        ASSERT_STATUS(scheduler->run());
    }

    void EFairServer::run() {
        if (scheduler.get() == nullptr)
            throw std::runtime_error("Scheduler is not initialized.");

        grpc::ServerBuilder builder;

        builder.AddListeningPort(address, grpc::InsecureServerCredentials());
        builder.RegisterService(this);

        server = builder.BuildAndStart();
        LOG(INFO) << "Server listening on " << address;

        server->Wait();
    }

    grpc::Status EFairServer::LoadModel(grpc::ServerContext *context, const efair::rpc::LoadModelRequest *request,
                                        efair::rpc::LoadModelResponse *response) {
        ModelID mid;
        Status s = scheduler->load_model(request->model_path(), request->model_profile_path(),
                                         request->eid(),request->frequency(), mid);

        if (s == Status::Succeed)
            response->set_success(true);
        else
            response->set_success(false);

        response->set_mid(mid);
        return grpc::Status::OK;
    }

    grpc::Status EFairServer::CreateEntity(grpc::ServerContext *context, const efair::rpc::CreateEntityRequest *request,
                                           efair::rpc::CreateEntityResponse *response) {
        EntityID eid;
        Status s = scheduler->create_entity(request->priority(), eid);

        if (s == Status::Succeed)
            response->set_success(true);
        else
            response->set_success(false);

        response->set_eid(eid);
        return grpc::Status::OK;
    }

    grpc::Status
    EFairServer::SetEntityPriority(grpc::ServerContext *context, const efair::rpc::SetEntityPriorityRequest *request,
                                   efair::rpc::SetEntityPriorityResponse *response) {
        Status s = scheduler->set_entity_priority(request->eid(), request->priority());

        if (s == Status::Succeed)
            response->set_success(true);
        else
            response->set_success(false);

        return grpc::Status::OK;
    }

    grpc::Status EFairServer::Infer(grpc::ServerContext *context, const efair::rpc::InferRequest *request,
                                    efair::rpc::InferResponse *response) {
        TaskID tid;
        Status s = scheduler->new_task(request->mid(), tid);

        if (s == Status::Succeed)
            response->set_success(true);
        else
            response->set_success(false);

        response->set_tid(tid);
        scheduler->wait_task(tid);

        return grpc::Status::OK;
    }

    efair::scheduler::EFairScheduler *EFairServer::get_scheduler() const {
        return scheduler.get();
    }

    void EFairServer::shutdown() {
        server->Shutdown();
        scheduler->shutdown();
    }

}   // namespace rpc
}   // namespace efair