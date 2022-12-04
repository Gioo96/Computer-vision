cmake_minimum_required(VERSION 2.8)
project(Lab_3)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} include)

add_library(mylib include/filter.h src/filter.cpp)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(mylib ${OpenCV_LIBS})
