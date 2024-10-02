
set(syndicateSOURCE sensorStack.cpp sensor.cpp camera.cpp)

set(syndicateHEADER sensorStack.h sensor.h camera.h)

add_library(syndicate ${syndicateSOURCE} ${syndicateHEADER})

target_include_directories(syndicate
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )

add_subdirectory(SpinnakerCamera)
add_subdirectory(OpenCVCamera)
add_subdirectory(RFEthernet)
add_subdirectory(MiniDSPMic)
add_subdirectory(SerialPort)
add_subdirectory(HardwareTrigger)

target_link_libraries (syndicate SpinnakerCamera OpenCVCamera RFEthernet MiniDSPMic SerialPort HardwareTrigger)