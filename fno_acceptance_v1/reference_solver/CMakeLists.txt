cmake_minimum_required(VERSION 3.16)
project(FrozenFNOAcceptanceDG LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Boost REQUIRED)

# The acceptance protocol uses one CPU thread. The copied numerical kernels are
# unchanged; omitting OpenMP simply selects their existing serial path.
add_executable(dg_acceptance src/acceptance_driver.cpp src/cole_cole.cpp)
target_include_directories(dg_acceptance PRIVATE include ${Boost_INCLUDE_DIRS})
target_compile_options(dg_acceptance PRIVATE -O3 -Wall -Wextra)

# Byte-for-byte copies of the existing driver and kernel, for manual regression.
add_executable(dg_original_regression original/src/dataset_driver.cpp original/src/cole_cole.cpp)
target_include_directories(dg_original_regression PRIVATE original/include ${Boost_INCLUDE_DIRS})
target_compile_options(dg_original_regression PRIVATE -O3 -Wall -Wextra)
