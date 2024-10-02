Installation
=====

To use syndicate, first install the following dependancies if your setup requires them:

**WINDOWS**
------------

*FLIR Cameras*

* Install FLIR Spinnaker SDK
* Add {path to dynamically linked libraries (.dll)} to Window's PATH variable.
* Ensure ${SPINNAKER_DIR} cmake variable is set correctly in the *cmake/FindSpinnaker.cmake* file


**Python Installation**
------------
.. code-block:: console

   (.venv) $ pip install syndicate