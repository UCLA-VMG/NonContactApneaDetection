# Syndicate of Sensors

A C++ repository for perceptual data capture from a syndicate of sensors.


Boost Installation Insturctions:

#run bootstrap.bat
bootstrap.bat
#Build non-default libraries
## it is important to note the correct toolset for this step
b2 -j8 toolset=msvc-14.3 address-model=64 architecture=x86 link=static threading=multi runtime-link=shared --build-type=minimal stage --stagedir=stage/x64