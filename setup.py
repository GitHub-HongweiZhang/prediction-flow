#! /usr/bin/env python


from setuptools import setup, find_packages
import prediction_flow


with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt') as f:
    INSTALL_REQUIRES = f.read().splitlines()

DISTNAME = 'prediction-flow'
DESCRIPTION = 'Deep-Learning based CTR models implemented by PyTorch'
MAINTAINER = 'Hongwei Zhang'
MAINTAINER_EMAIL = 'hw_zhang@outlook.com'
URL = 'https://github.com/GitHub-HongweiZhang/prediction-flow'
LICENSE = 'MIT'
VERSION = prediction_flow.__version__


def setup_package():
    setup(
        name=DISTNAME,
        packages=find_packages(),
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        url=URL,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        python_requires='>=3.6',
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
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
