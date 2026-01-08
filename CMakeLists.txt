cmake_minimum_required(VERSION 3.10)
project(pointgluecpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_ONNXRUNTIME "Use ONNX Runtime as inference engine" ON)
option(USE_OPENVINO "Use OpenVINO as inference engine" ON)
option(BUILD_EXAMPLES "Build examples for directory examples" ON)
option(BUILD_TESTING "Build tests for directory test" ON)

if(NOT USE_ONNXRUNTIME AND NOT USE_OPENVINO)
  message(FATAL_ERROR "Place choose one inference engine at last!")
endif()

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(TBB REQUIRED)

set(REQUIRED_LIBRARIES
    spdlog::spdlog
    ${OpenCV_LIBS}
    TBB::tbb
    utils
    npycpp)

set(POINT_SOURCE_FILES
    src/SuperPoint/SuperPoint.cc
    src/SuperPoint/SuperPointEngine.cc)
set(GLUE_SOURCE_FILES
    src/SuperGlue/SuperGlue.cc
    src/SuperGlue/SuperGlueEngine.cc)

set(UTILS_SOURCE_FILES
    src/utils/ViewerTools.cc)

if(USE_ONNXRUNTIME)
  add_definitions(-DONNXRUNTIMEENGINE)
  find_package(onnxruntime REQUIRED)
  list(APPEND REQUIRED_LIBRARIES onnxruntime::onnxruntime)
  list(APPEND POINT_SOURCE_FILES src/SuperPoint/ONNXRuntimeEngine.cc)
  list(APPEND GLUE_SOURCE_FILES src/SuperGlue/ONNXRuntimeEngine.cc)
  message(STATUS "onnxruntime infer engine is loaded")
endif()

if(USE_OPENVINO)
  add_definitions(-DOPENVINOENGINE)
  find_package(OpenVINO REQUIRED)
  list(APPEND REQUIRED_LIBRARIES openvino::frontend::onnx)
  list(APPEND POINT_SOURCE_FILES src/SuperPoint/OpenVINOEngine.cc)
  list(APPEND GLUE_SOURCE_FILES src/SuperGlue/OpenVINOEngine.cc)
  message(STATUS "openvino infer engine is loaded")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DDEBUG)
  add_definitions(-DDEBUG_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")
endif()

add_subdirectory(3rdparty/npycpp)

add_library(utils SHARED ${UTILS_SOURCE_FILES})
target_link_libraries(utils ${OpenCV_LIBS})
target_include_directories(
    utils PUBLIC
    ${OpenCV_INCLUDE_DIR}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_library(point SHARED ${POINT_SOURCE_FILES})
target_link_libraries(point ${REQUIRED_LIBRARIES})
target_include_directories(
    point PUBLIC
    ${OpenCV_INCLUDE_DIR}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_library(glue SHARED ${GLUE_SOURCE_FILES})
target_link_libraries(glue ${REQUIRED_LIBRARIES})
target_include_directories(
    glue PUBLIC
    ${OpenCV_INCLUDE_DIR}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(test)
endif(BUILD_TESTING)


# 安装路径配置
set(PACKAGE_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(PACKAGE_LIBS_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib)

install(
    TARGETS point glue utils
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${PACKAGE_LIBS_INSTALL_DIR}
    ARCHIVE DESTINATION ${PACKAGE_LIBS_INSTALL_DIR}
)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/incldue/
    DESTINATION ${PACKAGE_INCLUDE_INSTALL_DIR}
)

install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${PACKAGE_LIBS_INSTALL_DIR}/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

set(USE_ONNXRUNTIME_CONFIG ${USE_ONNXRUNTIME})
set(USE_OPENVINO_CONFIG ${USE_OPENVINO})
configure_file(
    ${PROJECT_SOURCE_DIR}/Config.cmake.in
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY
)

install(
    FILES
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION
    ${PACKAGE_LIBS_INSTALL_DIR}/cmake/${PROJECT_NAME}
)

set(header "=============================[CMAKE | ${PROJECT_NAME} | ${CMAKE_BUILD_TYPE}]=============================")
string(LENGTH "${header}" header_len)
string(REPEAT "=" ${header_len} separator)

message(STATUS ${header})
message(STATUS "${PROJECT_NAME}Configure: ")
message(STATUS "USE_ONNXRUNTIME:      ${USE_ONNXRUNTIME}")
message(STATUS "USE_OPENVINO:         ${USE_OPENVINO}")
message(STATUS "BUILD_EXAMPLES:       ${BUILD_EXAMPLES}")
message(STATUS "BUILD_TESTING:        ${BUILD_TESTING}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS ${separator})
