#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'matplotlib',
    'seaborn',
    'numpy',
    'pandas',
]

setup(
    name='kvae',
    version='0.1.1',
    description="Kalman Variational Auto-Encoder",
    long_description=readme,
    author="Simon Kamronn",
    author_email='simon@kamronn.dk',
    url='https://github.com/simonkamronn/kvae',
    packages=[
        'kvae',
    ],
    package_dir={'kvae': 'kvae'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='kalman variational auto-encoder lgssm state space',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
