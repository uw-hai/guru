cmake_minimum_required (VERSION 2.6)

function (AddTest name)
    add_executable(${name}_tests ${name}_tests.cpp)
    target_link_libraries(${name}_tests AIToolboxMDP AIToolboxPOMDP ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${name} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${name}_tests>)
endfunction (AddTest)


find_package(Boost 1.53 COMPONENTS unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
AddTest(util)
