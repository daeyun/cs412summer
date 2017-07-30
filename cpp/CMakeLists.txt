cmake_minimum_required(VERSION 3.5)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_VERBOSE_MAKEFILE 1)
set(BUILD_SHARED_LIBS 0)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

set(TARGET_PROTO_FILES
        ${CMAKE_BINARY_DIR}/proto/server.pb.cc
        ${CMAKE_BINARY_DIR}/proto/server.pb.h
        ${CMAKE_BINARY_DIR}/proto/server.grpc.pb.cc
        ${CMAKE_BINARY_DIR}/proto/server.grpc.pb.h
        )

set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third_party/nix_env)


find_package(OpenMP REQUIRED)
#find_package(Protobuf REQUIRED)
#message(STATUS "Protobuf_VERSION: ${Protobuf_VERSION}, path: ${PROTOBUF_INCLUDE_DIRS}, ${PROTOBUF_PROTOC_EXECUTABLE}")
#message(STATUS ${TARGET_PROTO_FILES})

#add_custom_command(
#        OUTPUT ${TARGET_PROTO_FILES}
#        COMMAND mkdir ARGS -p ${CMAKE_BINARY_DIR}/proto
#
#        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
#        ARGS
#        --proto_path=${CMAKE_SOURCE_DIR}/proto
#        --grpc_out=${CMAKE_BINARY_DIR}/proto
#        --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`
#        ${CMAKE_SOURCE_DIR}/proto/server.proto
#
#        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
#        ARGS
#        --proto_path=${CMAKE_SOURCE_DIR}/proto
#        --cpp_out=${CMAKE_BINARY_DIR}/proto
#        ${CMAKE_SOURCE_DIR}/proto/server.proto
#)
#
#add_library(protolib ${TARGET_PROTO_FILES})
#target_link_libraries(protolib protobuf)
#target_include_directories(protolib PUBLIC
#        ${PROTOBUF_INCLUDE_DIRS}
#        )
#target_link_libraries(protolib ${Protobuf_LIBRARIES})


if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()


include_directories(
        ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/src
        ${CMAKE_SOURCE_DIR}/third_party/nix_env/include
)
link_directories(${CMAKE_SOURCE_DIR}/third_party/nix_env/lib)


add_library(restbed STATIC IMPORTED)
set_property(TARGET restbed PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/third_party/install/restbed/library/librestbed.a)
set_property(TARGET restbed PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/third_party/install/restbed/include)


set(SOURCE_FILES main.cpp)
add_executable(cpp ${SOURCE_FILES})
target_link_libraries(cpp restbed ssl crypto glog re2 pthread)

