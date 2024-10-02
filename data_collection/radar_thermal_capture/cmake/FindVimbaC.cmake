# # Find the VimbaC library
# #
# #  VimbaC_FOUND        - True if VimbaC was found.
# #  VimbaC_LIBRARIES    - The libraries needed to use VimbaC
# #  VimbaC_INCLUDE_DIRS - Location of VimbaC.h

# unset(VimbaC_FOUND)
# unset(VimbaC_INCLUDE_DIRS)
# unset(VimbaC_LIBRARIES)

# find_path(VimbaC_INCLUDE_DIRS
#   NAMES
#   "VimbaC.h"
#   PATHS
#   "C:/Program Files/Allied Vision/Vimba_6.0/VimbaC/Include"
#   # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaC_Source/VimbaC/Include"
#   )
# message(STATUS "find_path result:")
# message(STATUS "${VimbaC_INCLUDE_DIRS}")

# find_library(VimbaC_LIBRARIES
#   NAMES
#   VimbaC.lib
#   PATHS
#   "C:/Program Files/Allied Vision/Vimba_6.0/VimbaC/Lib/Win64"
#   # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaC_Source/Build"
#   # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaC_Source/Build/VS2010/x64/Debug"
#   )

# message(STATUS "find_library result:")
# message(STATUS "${VimbaC_LIBRARIES}")

# if (VimbaC_FOUND)
#     message(STATUS "VimbaC include: ${VimbaC_INCLUDE_DIR}")
#     message(STATUS "VimbaC library: ${VimbaC_LIBRARY}")
#     set(VimbaC_INCLUDE_DIRS "${VimbaC_INCLUDE_DIR}")
#     set(VimbaC_LIBRARIES "${VimbaC_LIBRARY}")# "${VimbaC_LIBRARYD}")
# endif()

# if (VimbaC_INCLUDE_DIRS AND VimbaC_LIBRARIES)
#   message(STATUS "VimbaC found in the system")
#   set(VimbaC_FOUND 1)
# endif (VimbaC_INCLUDE_DIRS AND VimbaC_LIBRARIES)

##################################################################################################################

# Find the VimbaC library
#
#  VimbaC_FOUND        - True if VimbaC was found.
#  VimbaC_LIBRARIES    - The libraries needed to use VimbaC
#  VimbaC_INCLUDE_DIRS - Location of VimbaC.h

unset(VimbaC_FOUND)
unset(VimbaC_INCLUDE_DIRS)
unset(VimbaC_LIBRARIES)

find_path(VimbaC_INCLUDE_DIRS
  NAMES
  "VimbaC/Include/VimbaC.h"
  PATHS
  "C:/Program Files/Allied Vision/Vimba_6.0"
  # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaC_Source/VimbaC/Include"
  )
message(STATUS "find_path result:")
message(STATUS "${VimbaC_INCLUDE_DIRS}")

find_library(VimbaC_LIBRARIES
  NAMES
  "VimbaC/Lib/Win64/VimbaC.lib"
  PATHS
  "C:/Program Files/Allied Vision/Vimba_6.0"
  # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaC_Source/Build"
  # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaC_Source/Build/VS2010/x64/Debug"
  )

message(STATUS "find_library result:")
message(STATUS "${VimbaC_LIBRARIES}")

if (VimbaC_FOUND)
    message(STATUS "VimbaC include: ${VimbaC_INCLUDE_DIR}")
    message(STATUS "VimbaC library: ${VimbaC_LIBRARY}")
    set(VimbaC_INCLUDE_DIRS "${VimbaC_INCLUDE_DIR}")
    set(VimbaC_LIBRARIES "${VimbaC_LIBRARY}")# "${VimbaC_LIBRARYD}")
endif()

if (VimbaC_INCLUDE_DIRS AND VimbaC_LIBRARIES)
  message(STATUS "VimbaC found in the system")
  set(VimbaC_FOUND 1)
endif (VimbaC_INCLUDE_DIRS AND VimbaC_LIBRARIES)