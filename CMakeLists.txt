cmake_minimum_required(VERSION 3.16)
project(muti_thread)

set(CMAKE_CXX_STANDARD 20)
set(ngraph_DIR "/opt/intel/openvino_2024/runtime/cmake")
set(InferenceEngine_DIR "/opt/intel/openvino_2024/runtime/cmake")

find_package(OpenCV 4 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
find_package(fmt REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(
        #OpenVINO推理引擎的头文件
        /opt/intel/openvino_2024/runtime/include/ie/
        /opt/intel/openvino_2024/runtime/include/ngraph/
        /opt/intel/openvino_2024/runtime/include/openvino/
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_LIBRARY_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        

)
add_executable(${PROJECT_NAME} ${sourcefile} main.cpp)
target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        yaml-cpp
        /opt/intel/openvino_2024/runtime/lib/intel64/libopenvino.so
        openvino::frontend::onnx openvino::runtime
        fmt::fmt
)