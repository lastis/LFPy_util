from setuptools import setup, find_packages
setup(
    name = "LFPy_util",
    version = "1.0",
    packages = find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = ['NEURON >= 7.4','matplotlib >= 1.3.1',' LFPy >= 1.1.0'],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
        'hello': ['*.msg'],
    },

    # metadata for upload to PyPI
    author = "Daniel Marelius Bjoernstad",
    author_email = "neitakkikkebanan@gmail.com",
    description = "Extension package for LFPy and neuron.",
    license = "PSF",
)
