# # Find the VimbaCPP library
# #
# #  VimbaCPP_FOUND        - True if VimbaCPP was found.
# #  VimbaCPP_LIBRARIES    - The libraries needed to use VimbaCPP
# #  VimbaCPP_INCLUDE_DIRS - Location of VimbaCPP.h

# unset(VimbaCPP_FOUND)
# unset(VimbaCPP_INCLUDE_DIRS)
# unset(VimbaCPP_LIBRARIES)

# find_path(VimbaCPP_INCLUDE_DIRS
#   NAMES
#   "VimbaCPP.h"
#   PATHS
#   "C:/Program Files/Allied Vision/Vimba_6.0/VimbaCPP/Include"
#   # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaCPP_Source/VimbaCPP/Include"
#   )
# message(STATUS "find_path result:")
# message(STATUS "${VimbaCPP_INCLUDE_DIRS}")

# find_library(VimbaCPP_LIBRARIES
#   NAMES
#   VimbaCPP.lib
#   PATHS
#   "C:/Program Files/Allied Vision/Vimba_6.0/VimbaCPP/Lib/Win64"
#   "C:/Program Files/Allied Vision/Vimba_6.0/VimbaCPP/Bin/Win64"
#   # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaCPP_Source/Build"
#   # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaCPP_Source/Build/VS2010/x64/Debug"
#   )

# message(STATUS "find_library result:")
# message(STATUS "${VimbaCPP_LIBRARIES}")

# if (VimbaCPP_FOUND)
#     message(STATUS "VimbaCPP include: ${VimbaCPP_INCLUDE_DIR}")
#     message(STATUS "VimbaCPP library: ${VimbaCPP_LIBRARY}")
#     set(VimbaCPP_INCLUDE_DIRS "${VimbaCPP_INCLUDE_DIR}")
#     set(VimbaCPP_LIBRARIES "${VimbaCPP_LIBRARY}")# "${VimbaCPP_LIBRARYD}")
# endif()

# if (VimbaCPP_INCLUDE_DIRS AND VimbaCPP_LIBRARIES)
#   message(STATUS "VimbaCPP found in the system")
#   set(VimbaCPP_FOUND 1)
# endif (VimbaCPP_INCLUDE_DIRS AND VimbaCPP_LIBRARIES)

##################################################################################################################

# Find the VimbaCPP library
#
#  VimbaCPP_FOUND        - True if VimbaCPP was found.
#  VimbaCPP_LIBRARIES    - The libraries needed to use VimbaCPP
#  VimbaCPP_INCLUDE_DIRS - Location of VimbaCPP.h

unset(VimbaCPP_FOUND)
unset(VimbaCPP_INCLUDE_DIRS)
unset(VimbaCPP_LIBRARIES)

find_path(VimbaCPP_INCLUDE_DIRS
  NAMES
  "VimbaCPP/Include/VimbaCPP.h"
  PATHS
  "C:/Program Files/Allied Vision/Vimba_6.0"
  # "C:/Users/Public/Documents/Allied Vision/Vimba_6.0/VimbaCPP_Source/VimbaCPP/Include"
  )
message(STATUS "find_path result:")
message(STATUS "${VimbaCPP_INCLUDE_DIRS}")

find_library(VimbaCPP_LIBRARIES
  NAMES
  "VimbaCPP/Lib/Win64/VimbaCPP.lib"
  PATHS
  "C:/Program Files/Allied Vision/Vimba_6.0"
  )

message(STATUS "find_library result:")
message(STATUS "${VimbaCPP_LIBRARIES}")

if (VimbaCPP_FOUND)
    message(STATUS "VimbaCPP include: ${VimbaCPP_INCLUDE_DIR}")
    message(STATUS "VimbaCPP library: ${VimbaCPP_LIBRARY}")
    set(VimbaCPP_INCLUDE_DIRS "${VimbaCPP_INCLUDE_DIR}")
    set(VimbaCPP_LIBRARIES "${VimbaCPP_LIBRARY}")# "${VimbaCPP_LIBRARYD}")
endif()

if (VimbaCPP_INCLUDE_DIRS AND VimbaCPP_LIBRARIES)
  message(STATUS "VimbaCPP found in the system")
  set(VimbaCPP_FOUND 1)
endif (VimbaCPP_INCLUDE_DIRS AND VimbaCPP_LIBRARIES)