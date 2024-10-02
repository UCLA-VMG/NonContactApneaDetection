
set(RFEthernetSOURCE 
        RFEthernet.cpp)

set(RFEthernetHEADER
        RFEthernet.h)

add_library(RFEthernet ${RFEthernetSOURCE} ${RFEthernetHEADER})

target_include_directories(RFEthernet
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )

target_link_libraries(RFEthernet syndicate)