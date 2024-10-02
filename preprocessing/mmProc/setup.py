from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Library for Processing Multimodal Data'
LONG_DESCRIPTION = 'Methods include changing data format, facecropping, metadata generation and other image, audio and radar data processing functions'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="mmproc", 
        version=VERSION,
        author="Adnan Armouti",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package.
        
        keywords=['python', 'first package'],
        classifiers= [
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
        ]
)