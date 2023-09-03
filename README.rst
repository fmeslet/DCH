======================================================================================
DCH: A Deep Learning Approach To Universal Header Compression For The Internet of Things
======================================================================================



Summary
------------

DCH (Deep Compression Header) is a Deep Learning architecture used to compress network packet from network traffic. The model can learn to compress any packet without knowing the structure of its protocol. Thanks to a step of model transformation, DCH is embeddable on an IoT (Arduino) device. This work is published in `Conference name year <https://>`.



Processing
------------

The folder ``scripts/processing`` includes all the files that have allowed to process the data and to transform them to make the modeling. We use two types of scripts:

* The scripts to extract the packets, contained in the folder ``scripts/processing/packets/`` ;  
* The scripts to extract the flows, contained in the folder ``scripts/processing/flows/``.  


Packet level
^^^^^^^^^^^^^^

The scripts used for packet extraction are in the folder ``scripts/processing/packets/``. Their order of use is the following:  

* ``script_extraction_*_bit.py``: extracts a range of packets from a PCAP. The results are exported, for each packet, in a ``.csv`` file containing information about the characteristics (size, type of headers, source IP address, destination IP address, ...) and in a ``.npy`` file representing the header in binary format. In the case of COAP and MQTT, IPv4 headers are converted to IPv6.  
* ``script_reducer_arr_bit.py``: groups in a single numpy array all the packet ranges extracted in a ``.npy`` file, from the same protocol.  
* ``script_reducer_df.py``: groups in a single file all the packet ranges extracted in a ``.csv`` file, from the same protocol.  
* ``scapy_layers.py``:contains a set of classes to support protocols not supported by default by ``Scapy`` such as: IRC, IMAP, SMTP, POP, SIP, SSH, Telnet, FTP.  

Samples of the results obtained for each protocol after extraction of the packets are represented in the ``data/`` folder in the form of ``df_raw_*.csv``.  


Flow level
^^^^^^^^^^^^^^

The files ``df_raw_*.csv`` extracted for each protocol are used to identify the flows. The goal is to associate to each packet a flow identifier from the tuple (source IP address, destination IP address, source port and destination port). The extracted flows will identify the context packets for each packet to be compressed. 

The files used for flow extraction are inside the folder ``scripts/processing/packets/``. Their order of use is the following:

* ``script_flows_extraction_*.py``: identifies packets belonging to the same flow.  

Samples of the results obtained for each protocol after extraction of the flows are represented in the file ``data/process/`` in the form of files named ``df_process_*.csv``.



Training
------------

The files are present in the ``training`` folder for each associated level. Their order of use is the following:

* ``script_compressor_lossless_generator_*_training.py``: learns the pattern, which can be "baseline", "baseline naive" or DCH if nothing is specified. "baseline naive" corresponds to a model based on the frequency of appearance of patterns. "baseline" uses the same principle as "baseline naive" but integrates the notion of position.  

The script ``script_compressor_lossless_generator_baseline_training_reducer.py`` allows to aggregate the frequency tables of each table associated to each position. The folder ``models`` contains the models used for the training. 

The scripts named with the word ``interpretation`` are identical in their operation but are adapted to be able to work with artificial datasets used to understand the impact of DCH on different type of field structure in different configuration (context size, window size).



Tranformation
------------

The folder ``scripts/processing`` includes all the files that have allowed the model transformation. DCH in the form of a deep learning model cannot run on an IoT device. The model transformation allows to transform DCH into a table in order to embed it on an IoT device. The transformation uses the Occlusion Map.

The transformation step consists of two sub-steps:

* The application of the Occlusion Map ;  
* Table extraction.  

Scripts named with the word ``interpretation`` are identical in their operation but are adapted to work with artificial datasets used to understand the impact of DCH on different types of field structure in different configurations (context size, window size).  


Occlusion Map
^^^^^^^^^^^^^^

The Occlusion Map consists in applying a mask on each bit in input of DCH to measure the impact on the variability of the predicted bit. If the variaton is important, then, the masked bit has importance for the prediction. The measured variaton for each masked bit and the results are stored in ``df_FIELDS_*.csv``.

The scripts used for the Occlusion Map applications are the following:

* ``script_compressor_lossless_generator_*_occlusion_map_by_block.py``: applies a mask on each bit in input of DCH on a portion of the data. The impact of the variation of the output probability of DCH for each masked bit is saved in files named ``df_FIELDS_*.csv``. In order to reduce the memory footprint the operation is performed per block and a set of files is obtained as output.  
* ``script_compressor_lossless_generator_*_occlusion_map_by_block_reducer.py``: groups the obtained ``df_FIELDS_*.csv`` files into a single file.  

A sample of the output from these scripts is presented in the ``results/fields`` folder.


Table extraction
^^^^^^^^^^^^^^

The scripts used for the extraction of the tables are the following:

* ``script_compressor_lossless_generator_*_huffman_table_extraction.py``: extracts for each bit position and for each context size a table with the associated probability to get the value of a bit at 0 for each position in a file named ``df_HUFFMAN_LOSSLESS_*.csv``. ``arr_index_pos_HUFFMAN_*.npy`` gathers the position of the most important bits to use to determine the probability.
* ``script_compressor_lossless_generator_*_huffman_table_extraction_reducer.py``: combines the files ``df_HUFFMAN_LOSSLESS_*.csv`` and ``arr_index_pos_HUFFMAN_*.npy`` obtained in a single file.  

The tables obtained from these scripts are presented in the file ``models/huffman/``.
The numpy array named ``arr_index_pos_HUFFMAN_*.npy`` indicates the position of the bits to be extracted according to the context size and the position of the bit to be compressed.  
The files named ``df_HUFFMAN_LOSSLESS_*.csv`` show the probability of getting a 0 bit for each bit position and context size.  

Several files exist depending on the context size, the window size and the number of bits used. The chosen parameters are specified in the file name.
 


Evaluation
------------

We distinguish two levels of evaluation:

* **Offline**: the evaluation is done on models running only on a cluster or with a sufficiently powerful machine. This concerns DCH in the form of a deep learning model, without table transformation.
* **Online**: The evaluation is done on models that can be embedded directly on an IoT device. We find the "baseline naive" model and DCH trasnformed in table form.

The ``results/`` folder contains the results of the experiments performed. We find an associated folder for each model:

* ``results/baseline_naive/``: results obtained by the reference model used to compare the results.  
* ``results/deep_learning/``: results obtained by DCH, without transformation, according to different configurations.  
* ``results/huffman/``: results obtained after transformation of DCH into table according to different configurations.  
* ``results/ìnterpretation/``: results from the model interpretation. Artificial datasets are used to understand the impact of DCH on different types of field structures in different configurations (context size, window size).  

The ``Graphs_plot.ipynb`` notebook in the ``notebooks/`` directory is used to generate graphs for visualizing the results.  



Proof of concept
------------

In order to demonstrate the ability of DCH to run on an IoT device a Proof of Concept (POC) has been implemented..

The folder ``proof_of_concept/`` contains the files necessary for the implementation of the POC. Three folders are present:

* ``proof_of_concept/iot``: contains the code to be deployed on the IoT device.  
* ``proof_of_concept/code_generation``: contains the code to generate the compressed tables to be embedded on the IoT device.  
* ``proof_of_concept/computer``: contains the code to be deployed on the computer with which the IoT device will communicate.  

A file ``README.rst`` is present in the directory to explain the set up of the POC.



Requirements
------------

* Python 3.6.0  
* TensorFlow 2.4.1  
* Numpy 1.14.3  
* Pandas 0.22.0  
* Scapy 2.4.3  
* Scapy_ssl_tls 2.0.0  


Updates
-------

* Version 1.0.0  



Authors
-------

* **Fabien Meslet-Millet**  



Contributors
------------

*



LICENSE
-------

See the file "LICENSE" for information.
