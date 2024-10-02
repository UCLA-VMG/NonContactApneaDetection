
set(HardwareTriggerSOURCE 
        HardwareTrigger.cpp)

set(HardwareTriggerHEADER
        HardwareTrigger.h)

add_library(HardwareTrigger ${HardwareTriggerSOURCE} ${HardwareTriggerHEADER})

target_include_directories(HardwareTrigger
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )

target_link_libraries(HardwareTrigger PUBLIC syndicate)
