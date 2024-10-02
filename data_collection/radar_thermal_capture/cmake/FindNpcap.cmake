# Find the Npcap library
#
#  Npcap_FOUND        - True if Npcap was found.
#  Npcap_LIBRARIES    - The libraries needed to use Npcap
#  Npcap_INCLUDE_DIRS - Location of Npcap.h

unset(Npcap_FOUND)
unset(Npcap_INCLUDE_DIRS)
unset(Npcap_LIBRARIES)

set(Npcap_DIR "C:/Program\ Files/npcap-sdk-1.12")

find_path(Npcap_INCLUDE_DIRS NAMES
  pcap.h
  PATHS
  "${Npcap_DIR}/Include" 
  "${Npcap_DIR}/Include/pcap"
  )

if(X86)
	find_library(Npcap_LIBRARY NAMES Npcap_v140.lib
	  PATHS
	  "${Npcap_DIR}/lib/vs2015/"
	)
elseif(X64)
	find_library(Npcap_packet_LIBRARY NAMES Packet.lib
	  PATHS
	  "${Npcap_DIR}/Lib/x64/"
	)
	find_library(Npcap_wpcap_LIBRARY NAMES wpcap.lib
	  PATHS
	  "${Npcap_DIR}/Lib/x64/"
	)
endif()

if(Npcap_INCLUDE_DIRS)
	message(STATUS "Npcap Header Files Found: ${Npcap_INCLUDE_DIRS}")
else()
	message(STATUS "Npcap Header Files Not Found")
endif()

if(Npcap_packet_LIBRARY)
	message(STATUS "Npcap packet Library Found: ${Npcap_packet_LIBRARY}")
else() 
	message(STATUS "Npcap packet Library Not Found")
endif()

if(Npcap_wpcap_LIBRARY)
	message(STATUS "Npcap wpcap Library Found: ${Npcap_wpcap_LIBRARY}")
else() 
	message(STATUS "Npcap wpcap Library Not Found")
endif()

set(Npcap_LIBRARIES ${Npcap_packet_LIBRARY} ${Npcap_wpcap_LIBRARY})




if (Npcap_INCLUDE_DIRS AND Npcap_LIBRARIES)
  message(STATUS "Npcap found in the system")
  set(Npcap_FOUND 1)
endif (Npcap_INCLUDE_DIRS AND Npcap_LIBRARIES)