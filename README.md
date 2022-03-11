# INDIGOpy
Implementation of INDIGO (INferring Drug Interactions using chemoGenomics and Orthology) as a Python package.

## References to build package
* https://www.freecodecamp.org/news/build-your-first-python-package/
* https://www.tutorialsteacher.com/python/python-package
* https://www.blog.pythonlibrary.org/2021/09/23/python-101-how-to-create-a-python-package/

## Tentative repo tree structure: 
```
INDIGOpy
└───data
│   │   ecoli_chemogenomics.xlsx
│   │   mtb_transcriptomics.xlsx
└───docs
└───examples
│   │   ecoli_demo.ipynb
│   │   mtb_demo.ipynb
└───indigo
│   │   __init__.py
│   │   modeling.py
│   │   orthology.py
│   │   preprocessing.py
│   │   visualization.py
└───tests
│   │   test_modeling.py
│   │   test_preprocessing.py
│   │   test_visualization.py
|   LICENSE
|   README.md
|   setup.py
```

## To do: 
- [ ] Configure `LICENSE` file
- [ ] Configure `setup.py` file
- [ ] Create + populate `data/` directory
  - [ ] Deposit `data/ecoli_chemogenomics.xlsx` file
  - [ ] Deposit `data/mtb_transcriptomics.xlsx` file
- [ ] Create + populate `docs/` directory
- [ ] Create + populate `examples/` directory
  - [ ] Create + populate `examples/ecoli_demo.ipynb` notebook
  - [ ] Create + populate `examples/mtb_demo.ipynb` notebook
- [ ] Create + populate `indigo/` package
  - [ ] Configure `indigo/__init__.py` file
  - [ ] Create + populate `indigo/modeling.py` module
  - [ ] Create + populate `indigo/orthology.py` module
  - [ ] Create + populate `indigo/preprocessing.py` module
  - [ ] Create + populate `indigo/visualization.py` module
- [ ] Create + populate `tests/` directory 
  - [ ] Create + populate `tests/test_modeling.py` module
  - [ ] Create + populate `tests/test_preprocessing.py` module
  - [ ] Create + populate `tests/test_visualization.py` module
- [ ] Deploy + test package in [TestPyPi](https://test.pypi.org/)
- [ ] Deploy + publish package in [PyPi](https://pypi.org/)
