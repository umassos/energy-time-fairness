//
// Created by Qianlin Liang on 3/23/23.
//

#ifndef EFAIR_SERVER_H
#define EFAIR_SERVER_H

#include <grpc++/grpc++.h>

#include "scheduler/scheduler.h"
#include "efair.grpc.pb.h"

namespace efair {
namespace rpc {

    class EFairServer final : public EFairService::Service {
    public:
        EFairServer(std::string address, efair::scheduler::EFairScheduler *scheduler_ptr);
        virtual ~EFairServer() = default;
        void run();
        void shutdown();
        efair::scheduler::EFairScheduler* get_scheduler() const;

    private:
        grpc::Status LoadModel(grpc::ServerContext *context, const efair::rpc::LoadModelRequest *request,
                               efair::rpc::LoadModelResponse *response) override;

        grpc::Status CreateEntity(grpc::ServerContext *context, const efair::rpc::CreateEntityRequest *request,
                                  efair::rpc::CreateEntityResponse *response) override;

        grpc::Status
        SetEntityPriority(grpc::ServerContext *context, const efair::rpc::SetEntityPriorityRequest *request,
                          efair::rpc::SetEntityPriorityResponse *response) override;

        grpc::Status Infer(grpc::ServerContext *context, const efair::rpc::InferRequest *request,
                           efair::rpc::InferResponse *response) override;

        std::string address;
        std::unique_ptr<efair::scheduler::EFairScheduler> scheduler;
        std::unique_ptr<grpc::Server> server;

    };

}   // namespace rpc
}   // namespace efair

#endif //EFAIR_SERVER_H


