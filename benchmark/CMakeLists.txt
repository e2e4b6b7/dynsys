cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_COMPILER g++-10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pipe -Ofast -Wall -Wextra")
message(${CMAKE_CXX_FLAGS})

project(DynamicSystemsBenchmark)

include_directories(../include/)

find_package(Qt5Widgets REQUIRED)
find_package(benchmark REQUIRED)

add_executable(DynamicSystemsBenchmark
    benchAll.cpp
    ../src/DynamicSystemParser/DynamicSystemParser.cpp
    ../src/Parser/Parser.cpp
    ../src/Parser/Lexer.cpp
)

qt5_use_modules(DynamicSystemsBenchmark Widgets)

target_link_libraries(DynamicSystemsBenchmark
    ${QT_LIBRARIES}
    benchmark::benchmark
)