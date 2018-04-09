cmake_minimum_required(VERSION 3.9)
project(SinewaveValidation)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)
find_package(Boost COMPONENTS chrono random date_time REQUIRED)

add_subdirectory(minusDarwin)
add_subdirectory(synthSignal)
add_subdirectory(DSPFilters/shared/DSPFilters)

add_executable(MethodsComparison methodsComparison.cpp ProposedMethodology.cpp ProposedMethodology.hpp)
target_include_directories(MethodsComparison PUBLIC minusDarwin synthSignal DSPFilters/shared/DSPFilters ${OpenCL_INCLUDE_DIR} ${Boost_INCLUDE_DIR})
target_link_libraries(MethodsComparison PUBLIC MinusDarwin SynthSignal DSPFilters OpenCL::OpenCL ${Boost_LIBRARIES} -lpthread -lm)

add_executable(FilterTestbench FilterTestbench.cpp)
target_include_directories(FilterTestbench PUBLIC minusDarwin synthSignal DSPFilters/shared/DSPFilters ${OpenCL_INCLUDE_DIR} ${Boost_INCLUDE_DIR})
target_link_libraries(FilterTestbench PUBLIC MinusDarwin SynthSignal DSPFilters OpenCL::OpenCL ${Boost_LIBRARIES} -lpthread -lm)
