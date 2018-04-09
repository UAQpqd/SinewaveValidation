cmake_minimum_required(VERSION 3.9)
project(SinewaveValidation)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)
find_package(Boost COMPONENTS chrono random date_time REQUIRED)

add_subdirectory(minusDarwin)
add_subdirectory(synthSignal)

add_executable(MethodsComparison methodsComparison.cpp ProposedMethodology.cpp ProposedMethodology.hpp)
target_include_directories(MethodsComparison PUBLIC minusDarwin synthSignal ${OpenCL_INCLUDE_DIR} ${Boost_INCLUDE_DIR})
target_link_libraries(MethodsComparison PUBLIC MinusDarwin SynthSignal OpenCL::OpenCL ${Boost_LIBRARIES} -lpthread -lm)
