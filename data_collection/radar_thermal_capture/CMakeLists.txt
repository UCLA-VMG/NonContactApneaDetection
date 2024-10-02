#DEV Notes:
#cmake --build ./ --target build_all --config Debug
#cmake --build ./ --target {custom .exe} --config Debug

cmake_minimum_required(VERSION 3.0.0)
project(syndicate VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 23)
include(CTest)
enable_testing()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)")
    set (X86 TRUE)
    set (X64 FALSE)
    message(STATUS "HOST Processor: ${CMAKE_HOST_SYSTEM_PROCESSOR}. Switching to X86 compilation.")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(x64)|(X64)|(amd64)|(AMD64)")
    set (X64 TRUE)
    set (X86 FALSE)
    message(STATUS "HOST Processor: ${CMAKE_HOST_SYSTEM_PROCESSOR}. Switching to X64 compilation.")
else()
    message(SEND_ERROR "HOST Processor, ${CMAKE_HOST_SYSTEM_PROCESSOR}, not compatible." )
endif ()

#Set Useful Paths
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#Find packeges
message(STATUS "${CMAKE_MODULE_PATH}")
find_package(Spinnaker REQUIRED)
find_package(Npcap REQUIRED)
find_package(OpenCV REQUIRED)

# set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost COMPONENTS thread filesystem REQUIRED)

#Include Path Libraries
include_directories("${PROJECT_SOURCE_DIR}/include"
                    "${PROJECT_SOURCE_DIR}/src"
                    SYSTEM ${Spinnaker_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIR}
                    ${Npcap_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    )

#Include Libraries
add_subdirectory(Syndicate)
set (LIBS ${Spinnaker_LIBRARIES} ${Boost_LIBRARIES} ${Npcap_LIBRARIES} ${OpenCV_LIBRARIES} ${WINDOWS_SDK} syndicate )

#Generate Execs
add_executable(capture_config ${PROJECT_SOURCE_DIR}/scripts/main.cpp)
target_link_libraries(capture_config ${LIBS})

#Targets
ADD_CUSTOM_TARGET(build_all)
ADD_DEPENDENCIES(build_all capture_config)

#Library Linking
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
