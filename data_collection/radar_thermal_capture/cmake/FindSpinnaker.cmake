# Find the Spinnaker library
#
#  Spinnaker_FOUND        - True if Spinnaker was found.
#  Spinnaker_LIBRARIES    - The libraries needed to use Spinnaker
#  Spinnaker_INCLUDE_DIRS - Location of Spinnaker.h

unset(Spinnaker_FOUND)
unset(Spinnaker_INCLUDE_DIRS)
unset(Spinnaker_LIBRARIES)

set(SPINNAKER_DIR "C:/Program\ Files/FLIR\ Systems/Spinnaker")

find_path(Spinnaker_INCLUDE_DIRS NAMES
  Spinnaker.h
  PATHS
  "${SPINNAKER_DIR}"
  "${SPINNAKER_DIR}/include/"
  )

if(X86)
	find_library(Spinnaker_LIBRARY NAMES Spinnaker_v140.lib
	  PATHS
	  "${SPINNAKER_DIR}/lib/vs2015/"
	)
elseif(X64)
	find_library(Spinnaker_LIBRARY NAMES Spinnaker_v140.lib
	  PATHS
	  "${SPINNAKER_DIR}/lib64/vs2015/"
	)
endif()

if(Spinnaker_INCLUDE_DIRS)
	message(STATUS "Spinnaker Header Files Found: ${Spinnaker_INCLUDE_DIRS}")
else()
	message(STATUS "Spinnaker Header Files Not Found")
endif()

if(Spinnaker_LIBRARY)
	message(STATUS "Spinnaker Library Found: ${Spinnaker_LIBRARY}")
else() 
	message(STATUS "Spinnaker Library Not Found")
endif()

set(Spinnaker_LIBRARIES ${Spinnaker_LIBRARY})




if (Spinnaker_INCLUDE_DIRS AND Spinnaker_LIBRARIES)
  message(STATUS "Spinnaker found in the system")
  set(Spinnaker_FOUND 1)
endif (Spinnaker_INCLUDE_DIRS AND Spinnaker_LIBRARIES)