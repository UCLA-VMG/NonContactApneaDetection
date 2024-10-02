
set(SerialPortSOURCE 
        SerialPort.cpp)

set(SerialPortHEADER
        SerialPort.h)

add_library(SerialPort ${SerialPortSOURCE} ${SerialPortHEADER})

target_include_directories(SerialPort
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )

target_link_libraries(SerialPort PUBLIC syndicate)
