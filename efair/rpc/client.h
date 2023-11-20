//
// Created by Qianlin Liang on 3/24/23.
//

#ifndef EFAIR_CLIENT_H
#define EFAIR_CLIENT_H

#include <memory>
#include <string>

#include "util/common.h"
#include "efair.grpc.pb.h"

namespace efair {
namespace rpc {
    class EFairClient {
    public:
        EFairClient(std::string address, std::string model_path, std::string model_profile_path, std::string freq,
                    int priority);
        ~EFairClient() = default;

        bool infer();

    private:
        std::unique_ptr<efair::rpc::EFairService::Stub> stub;
        ModelID mid;
        EntityID eid;
        int priority;
        std::string freq;
    };
}   // namespace rpc
}   // namespace efair
#endif //EFAIR_CLIENT_H
