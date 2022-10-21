# INDIGOpy
Implementation of INDIGO (INferring Drug Interactions using chemoGenomics and Orthology) as a Python package.

[![license badge](https://img.shields.io/badge/License-GNU%20GPL-blue)](https://github.com/sriram-lab/INDIGOpy/blob/main/LICENSE)
[![DOI badge](https://badgen.net/badge/DOI/10.15252%2Fmsb.20156777/blue)](https://doi.org/10.15252/msb.20156777)

# License
Released via GPL GNU License  
&copy; 2022 The Regents of the University of Michigan  
Chandrasekaran Research Group - https://systemsbiologylab.org/  
Contact: csriram@umich.edu  

# Installation and execution  
Refer to the official [INDIGOpy documentation]() for installation instructions and example code.  

## References to build package
* https://www.freecodecamp.org/news/build-your-first-python-package/
* https://www.tutorialsteacher.com/python/python-package
* https://www.blog.pythonlibrary.org/2021/09/23/python-101-how-to-create-a-python-package/

# Repository structure: 
```
INDIGOpy
└───docs
└───examples
│   │   indigopy_demo.ipynb
│   │   prepare_data_from_xlsx.ipynb
│   │   sample_data.xlsx
└───indigopy
│   │   __init__.py
│   │   core.py
│   │   sample_data.pkl
└───tests
│   │   __init__.py
│   │   test_core.py
└───venv
|   LICENSE
|   README.md
|   setup.py
```

## To do: 
- [X] Configure `LICENSE` file
- [ ] Configure `setup.py` file
- [X] Create + populate `docs/` directory using Sphinx
- [X] Create + populate `examples/` directory
- [X] Create + populate `indigopy/` package
- [X] Create + populate `tests/` directory 
- [ ] Deploy + test package in [TestPyPi](https://test.pypi.org/)
- [ ] Deploy + publish package in [PyPi](https://pypi.org/)
