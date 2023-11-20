//
// Created by tx2 on 2/20/23.
//

#ifndef EFAIR_COMMON_H
#define EFAIR_COMMON_H

#include <glog/logging.h>
#include <iostream>

#define ASSERT(condition)\
     do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << ":" << __LINE__ << std::endl; \
            std::terminate(); \
        } \
    } while (false)

#define ASSERT_STATUS(fn) ASSERT(fn == efair::Status::Succeed)

#define RETURN_STATUS(fn) \
{\
    Status s = fn;\
    if (s != Status::Succeed) { \
        LOG(ERROR) << #fn " error" << __FILE__ << ":" << __LINE__; \
        return s; \
    }\
}

namespace efair {
    enum Status {
        Succeed,
        Fail,
        NotFound,
        NoPrivilege
    };

    typedef size_t MicroSeconds;
    typedef size_t MicroJoule;
    typedef size_t MilliWatt;

    typedef size_t ModelID;
    typedef size_t EntityID;
    typedef size_t TaskID;

    typedef size_t KernelIdx;

    typedef double VRuntime;
    typedef int Priority;
}

#endif //EFAIR_COMMON_H