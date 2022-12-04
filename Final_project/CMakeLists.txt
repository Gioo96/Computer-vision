cmake_minimum_required(VERSION 2.8)
project(Final_project)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} include)
add_library(mylib include/detection_segmentation.h src/cpp/detection_segmentation.cpp)

add_executable(${PROJECT_NAME} src/cpp/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(mylib ${OpenCV_LIBS})
