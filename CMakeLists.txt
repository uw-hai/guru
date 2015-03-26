cmake_minimum_required (VERSION 2.6)

project (WorkLearn)

ADD_DEFINITIONS(
    -std=c++11 # Or -std=c++0x
    -Wall
    -Wextra
    # Other flags
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# All executables are tests.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test/bin)

enable_testing()

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        /home/jbragg/dev/ai-toolbox/include
)
link_directories(/home/jbragg/dev/ai-toolbox/build)
add_subdirectory (${PROJECT_SOURCE_DIR}/src)
add_subdirectory (${PROJECT_SOURCE_DIR}/test)
