======================================================================================
DCH: Proof of Concept
======================================================================================



Summary
------------

The aim of this dossier is to demonstrate the implementation of DCH for real-time network packet compression. The scripts are deployed using the Arduino IDE on a ``TTGO LoRa32 SX1276``. The protocol used for communication is WiFi. 

There are three folders that bring together different elements:

* ``proof_of_concept/code_generation/``: contains a notebook used for code generation.  
* ``proof_of_concept/computer/``: contains the scripts to be run on the computer that will act as the IoT gateway.  
* ``proof_of_concept/iot/``: contient les scripts a déployer sur l'equipemetn IoT qui permettront d'envoyer l'information à la passerelle.  



Code generation
------------

The code to be executed on the computer is contained in the ``proof_of_concept/code_generation/`` folder.

After execution of the cells, two files are generated: ``frequency_table.cpp`` and ``frequency_table.h``. These files can be placed directly in the Arduino project folder. These files can be placed directly in the Arduino project folder.



Computer
------------

The code to be run on the computer is contained in the ``proof_of_concept/computer/`` folder.

In order to function, the computer must be placed in the Wifi access point mode with the name ``dell`` (without password). Other names can be used, but files must be modified accordingly. This plays the role of an IoT gateway.

The following scripts must be run in order:

1. ``script_init.sh``: authorizes access to ``/dev/ttyUSB0`` (where the IoT device is connected).  
2. ``script_init_interface.sh``: creates the ``mon1`` network interface, which monitors traffic between the IoT device and the gateway.  
3. ``script_reception.py``: starts listening to the ``mon1`` interface and decompressing messages sent by the IoT device.  

**The code must be executed before the code is deployed on the IoT device, to ensure proper context synchronization.**  



IoT device
------------


The code to be deployed on the IoT device is contained in the ``proof_of_concept/iot/`` folder. Two folders are contained in the directory:

* ``Sender/``: contains the program to be deployed on the IoT device to perform compression and transmission.  
* ``Sender_performance/``: contains the program to be deployed on the IoT device to perform compression and transmission. Additional code is integrated for performance measurements.  

The ``proof_of_concept/iot/Sender/`` and ``proof_of_concept/iot/Sender_performance/`` directories contain generated files such as: ``frequency_table.cpp``, ``frequency_table.h``.

The program connects to the "Dell" access point (launched on the gateway computer) and transmits messages regularly. As long as the equipment is not connected to the access point, no messages are transmitted.

**The code must be executed after the code is deployed on the gateway, to ensure proper context synchronization.**


