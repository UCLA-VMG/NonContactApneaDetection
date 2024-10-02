
set(OpenCVSOURCE 
        OpenCVCamera.cpp)

set(OpenCVHEADER
        OpenCVCamera.h)

add_library(OpenCVCamera ${OpenCVSOURCE} ${OpenCVHEADER})

target_include_directories(OpenCVCamera
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )

  target_link_libraries(OpenCVCamera syndicate)
