"""A setuptools for the INDIGOpy module. 

Note: adapted from a setuptools template by setuptools.

"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # Project name that determines how users can install this project, e.g.:
    # $ pip install sampleproject
    name="indigopy-sriram-lab",  # Required
    # 
    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    version="1.1.0",  # Required
    # 
    # This is a one-line description or tagline of what your project does.
    description="The Python implementation of INDIGO",  # Optional
    # 
    # This is an optional longer description of your project.
    long_description=long_description,  # Optional
    # 
    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    long_description_content_type="text/markdown",  # Optional (see note above)
    # 
    # This should be a valid link to your project's main homepage.
    url="https://github.com/sriram-lab/INDIGOpy",  # Optional
    # 
    # This should be your name or the name of the organization which owns the project.
    author="Chandrasekaran Lab",  # Optional
    # 
    # This should be a valid email address corresponding to the author listed above.
    author_email="csriram@umich.edu",  # Optional
    # 
    # Classifiers help users find your project by categorizing it.
    classifiers=[  # Optional
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # 
    # This field adds keywords for your project.
    keywords="INDIGO, machine learning, combination therapy",  # Optional
    # 
    # This determines how to find package directories.
    packages=find_packages(),  # Required
    #  
    # Incude package data.
    include_package_data=True, 
    # 
    # Specify which Python versions you support.
    python_requires=">=3.8, <4",
    # 
    # This field lists other packages that your project depends on to run.
    install_requires=["numpy", "pandas", "tqdm"],  # Optional
    # 
    # Data files to be included in your packages.
    package_data={  # Optional
        "": ["*.pkl", "*.png", "*.xlsx"],
    },
    # 
    # List additional URLs that are relevant to your project as a dict.
    project_urls={  # Optional
        "Bug Reports": "https://github.com/sriram-lab/INDIGOpy/issues",
        "Source": "https://github.com/sriram-lab/INDIGOpy/",
        "Lab Website": "https://systemsbiologylab.org/", 
    },
)
