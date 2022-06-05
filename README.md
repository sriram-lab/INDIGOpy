# INDIGOpy
Implementation of INDIGO (INferring Drug Interactions using chemoGenomics and Orthology) as a Python package.

## References to build package
* https://www.freecodecamp.org/news/build-your-first-python-package/
* https://www.tutorialsteacher.com/python/python-package
* https://www.blog.pythonlibrary.org/2021/09/23/python-101-how-to-create-a-python-package/

## Tentative repo tree structure: 
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
- [ ] Configure `LICENSE` file
- [ ] Configure `setup.py` file
- [X] Create + populate `docs/` directory using Sphinx
- [X] Create + populate `examples/` directory
- [X] Create + populate `indigopy/` package
- [X] Create + populate `tests/` directory 
- [ ] Deploy + test package in [TestPyPi](https://test.pypi.org/)
- [ ] Deploy + publish package in [PyPi](https://pypi.org/)
