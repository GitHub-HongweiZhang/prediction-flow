#! /usr/bin/env python


from setuptools import setup
import prediction_flow


with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

DISTNAME = 'prediction-flow'
DESCRIPTION = ''
MAINTAINER = 'Hongwei Zhang'
MAINTAINER_EMAIL = ''
URL = 'https://github.com/GitHub-HongweiZhang/prediction-flow'
LICENSE = 'MIT'
VERSION = prediction_flow.__version__


def setup_package():
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        url=URL,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        python_requires='>=3.6',
        install_requires=[
            'numpy>=1.16.0',
            'pandas==0.24.2',
            'torch>=1.1.0',
            'tqdm>=4.32.0',
            'scikit-learn>=0.20.0',
            'h5py'
        ],
        classifiers=(
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
        ),
        license=LICENSE,
        keywords=[
            'torch', 'ctr prediction', 'deep learning',
            'deepfm', 'din', 'dnn', 'deep neural network']
    )


if __name__ == '__main__':
    setup_package()
