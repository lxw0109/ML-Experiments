# Machine Learning Experiments
Several small and easy projects on Machine Learning for beginners.

## Requirements
All the code in this project is implemented in [Python3.6+](https://www.python.org/downloads/).  
All essential packages for this project are listed in `requirements.txt`, you can install them by 
`pip install -r requirements.txt -i https://pypi.douban.com/simple/`  
[Anaconda](https://docs.anaconda.com/anaconda/) or [virtualenv + virtualenvwrapper](http://www.jianshu.com/p/44ab75fbaef2) are strongly recommended to manage your Python environments.

## Notes
1. Each directory in this project contains the (input & output) data(`data/`), implementations(`src/`) and corresponding documents(`docs/`) of a single small project.  
The structure of each directory in this project is as follows(taking `MNIST` as an example):  
```bash
MNIST/
├── data/
|   ├── input/
|   └── output/
├── docs
└── src
    ├── 1_mnist_tensorflow.py
    └── 2_mnist_keras.py
```
The code files in `src/` are organized in a sequential number as prefix.  
2. Brief instructions for each directory are as follows.
 
 | directory(project) | instruction |
 | :--- | :--- |
 | MNIST | training and recognizing handwritten digits |
 | CAPTCHA | training letters in CAPTCHA images and breaking easy CAPTCHA systems |
