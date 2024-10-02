##################################################################################################################

# Find the VimbaImageTransform library
#
#  VimbaImageTransform_FOUND        - True if VimbaImageTransform was found.
#  VimbaImageTransform_LIBRARIES    - The libraries needed to use VimbaImageTransform
#  VimbaImageTransform_INCLUDE_DIRS - Location of VimbaImageTransform.h

unset(VimbaImageTransform_FOUND)
unset(VimbaImageTransform_INCLUDE_DIRS)
unset(VimbaImageTransform_LIBRARIES)

find_path(VimbaImageTransform_INCLUDE_DIRS
  NAMES
  "VimbaImageTransform/Include/VmbTransform.h"
  PATHS
  "C:/Program Files/Allied Vision/Vimba_6.0/"
  # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaImageTransform_Source/VimbaImageTransform/Include"
  )
message(STATUS "find_path result:")
message(STATUS "${VimbaImageTransform_INCLUDE_DIRS}")

find_library(VimbaImageTransform_LIBRARIES
  NAMES
  "VimbaImageTransform/Lib/Win64/VimbaImageTransform.lib"
  PATHS
  "C:/Program Files/Allied Vision/Vimba_6.0/"
  )

message(STATUS "find_library result:")
message(STATUS "${VimbaImageTransform_LIBRARIES}")

if (VimbaImageTransform_FOUND)
    message(STATUS "VimbaImageTransform include: ${VimbaImageTransform_INCLUDE_DIR}")
    message(STATUS "VimbaImageTransform library: ${VimbaImageTransform_LIBRARY}")
    set(VimbaImageTransform_INCLUDE_DIRS "${VimbaImageTransform_INCLUDE_DIR}")
    set(VimbaImageTransform_LIBRARIES "${VimbaImageTransform_LIBRARY}")# "${VimbaImageTransform_LIBRARYD}")
endif()

if (VimbaImageTransform_INCLUDE_DIRS AND VimbaImageTransform_LIBRARIES)
  message(STATUS "VimbaImageTransform found in the system")
  set(VimbaImageTransform_FOUND 1)
endif (VimbaImageTransform_INCLUDE_DIRS AND VimbaImageTransform_LIBRARIES)