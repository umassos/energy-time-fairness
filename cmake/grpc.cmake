cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 17)
set(protobuf_MODULE_COMPATIBLE TRUE)

find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

include_directories("${gRPC_DIR}/include")

find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
find_program(_PROTOBUF_PROTOC protoc)

get_filename_component(efair_proto "${CMAKE_CURRENT_SOURCE_DIR}/efair/proto/efair.proto" ABSOLUTE)
get_filename_component(efair_proto_path "${efair_proto}" PATH)

set(GRPC_GENERATED_PATH "${CMAKE_CURRENT_SOURCE_DIR}/efair/rpc")
set(efair_proto_srcs "${GRPC_GENERATED_PATH}/efair.pb.cc")
set(efair_proto_hdrs "${GRPC_GENERATED_PATH}/efair.pb.h")
set(efair_grpc_srcs "${GRPC_GENERATED_PATH}/efair.grpc.pb.cc")
set(efair_grpc_hdrs "${GRPC_GENERATED_PATH}/efair.grpc.pb.h")
add_custom_command(
        OUTPUT "${efair_proto_srcs}" "${efair_proto_hdrs}" "${efair_grpc_srcs}" "${efair_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${GRPC_GENERATED_PATH}"
             --cpp_out "${GRPC_GENERATED_PATH}"
             -I "${efair_proto_path}"
             --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
             "${efair_proto}"
        DEPENDS "${efair_proto}"
)

add_library(libefair_grpc_proto
        ${efair_grpc_srcs}
        ${efair_grpc_hdrs}
        ${efair_proto_srcs}
        ${efair_proto_hdrs}
        )
target_link_libraries(libefair_grpc_proto
        gRPC::grpc++
        ${Protobuf_LIBRARIES}
        )
