
set(SpinnakerSOURCE 
        SpinnakerCamera.cpp)

add_library(SpinnakerCamera ${SpinnakerSOURCE})

target_include_directories(SpinnakerCamera
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )

  target_link_libraries(SpinnakerCamera syndicate)
