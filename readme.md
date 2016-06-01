NEURON cannot be correctly installed from pip. Version 7.4 is needed with access
to commands such as nrnivmodl to compile mod files. 
This requires a myriad of libraries to compile correctly. 
Please ensure that Neuron runs correctly with nrnivmodl.

Install with :
	$ python setup.py install

Some packages do not install automatically
in virtualenv.
These include:
	matplotlib
	numpy
	scipy

Requires many latex packages for plotting. All are included
in texlive-full.
	$ sudo apt-get install texlive-full

Installing NEURON on Linux with Python:
1. Download and install .deb files from their page.
2. Append python libraries to PYTHONPATH.
    export PYTHONPATH=/usr/local/nrn/lib/python/:$PYTHONPATH
3. ???
4. Profit.
