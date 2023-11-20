cmake_minimum_required(VERSION 3.24)
project(eFair)

set(CMAKE_CXX_STANDARD 17)

# Add TVM build
include("${CMAKE_SOURCE_DIR}/cmake/config.cmake")
add_subdirectory("3rdparty/tvm")

add_definitions(-DMODEL_DIR="${CMAKE_CURRENT_LIST_DIR}/models")
add_definitions(-DSAMPLE_DIR="${CMAKE_CURRENT_LIST_DIR}/samples")
add_definitions(-DSERVER_ADDRESS="0.0.0.0:10086")
add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

aux_source_directory("${PROJECT_SOURCE_DIR}/efair/util" efair_util_src)
aux_source_directory("${PROJECT_SOURCE_DIR}/efair/executor" efair_executor_src)
aux_source_directory("${PROJECT_SOURCE_DIR}/efair/scheduler" efair_scheduler_src)
aux_source_directory("${PROJECT_SOURCE_DIR}/efair/rpc" efair_rpc_src)


find_package(Boost REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)

include("${CMAKE_SOURCE_DIR}/cmake/grpc.cmake")

# Set up TVM

#if (NOT DEFINED ENV{TVM_ROOT})
#    message(FATAL_ERROR, "Environment variable TVM_ROOT is not defined. ")
#endif ()

set(TVM_ROOT "3rdparty/tvm")

include_directories("${TVM_ROOT}/include")
include_directories("${TVM_ROOT}/3rdparty/dmlc-core/include")
include_directories("${TVM_ROOT}/3rdparty/dlpack/include")



#add_library(libtvm_runtime SHARED IMPORTED GLOBAL)
#
#if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#    set_target_properties(libtvm_runtime PROPERTIES IMPORTED_LOCATION "${TVM_ROOT}/build/libtvm_runtime.dylib")
#else()
#    set_target_properties(libtvm_runtime PROPERTIES IMPORTED_LOCATION "${TVM_ROOT}/build/libtvm_runtime.so")
#endif ()


include_directories("${PROJECT_SOURCE_DIR}/efair")
include_directories("${Boost_INCLUDE_DIRS}")


add_library(libefair_executor ${efair_executor_src})
target_link_libraries(libefair_executor
        tvm_runtime
        glog::glog          # Remove this to avoid re-define warning
        "${Boost_LIBRARIES}"
        )

add_library(libefair_util ${efair_util_src})
target_link_libraries(libefair_util
        glog::glog
        )

add_library(libefair_scheduler ${efair_scheduler_src})
target_link_libraries(libefair_scheduler
        libefair_executor
        libefair_util
        )

add_library(libefair_rpc ${efair_rpc_src} ${efair_grpc_srcs} ${efair_grpc_hdrs} ${efair_proto_srcs} ${efair_proto_hdrs})
target_link_libraries(libefair_rpc
        libefair_scheduler
        libefair_grpc_proto)

add_executable(efair_example efair/example/example.cpp)
target_link_libraries(efair_example
        libefair_executor
        libefair_util
        )

add_executable(efair_unittest efair/test/test.cpp)
target_link_libraries(efair_unittest
        libefair_scheduler
        libefair_executor
        ${GTEST_BOTH_LIBRARIES}
        pthread
        )

add_executable(profileDNN efair/profiler/profile_dnn.cpp)
target_link_libraries(profileDNN
        libefair_executor
        libefair_util
        )

add_executable(run_server efair/example/run_server.cpp)
target_link_libraries(run_server
        libefair_rpc
        )

add_executable(run_client efair/example/run_client.cpp efair/rpc/client.h efair/rpc/client.cpp)
target_link_libraries(run_client
        libefair_grpc_proto
        )
