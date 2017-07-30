# Needs to make sure the Nix symlinks are accessible.
#set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third_party/install/protobuf)
#find_package(Protobuf 3.2.0 EXACT REQUIRED)
find_package(Protobuf REQUIRED)

message(STATUS "Protobuf_VERSION: ${Protobuf_VERSION}, path: ${PROTOBUF_INCLUDE_DIRS}, ${PROTOBUF_PROTOC_EXECUTABLE}")

message(STATUS ${TARGET_PROTO_FILES})


