#!/usr/bin/env python

from setuptools import setup, Extension

try:
    import Cython
except ImportError:
    raise ImportError('''
Cython is required for building this package. Please install using

    pip install cython

or upgrade to a recent PIP release.
''')


with open('README.md') as f:
    long_description = f.read()


setup(
    name='PyMUMPS',
    version='0.3.2',
    description='Python bindings for MUMPS, a parallel sparse direct solver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Bradley M. Froehle',
    author_email='brad.froehle@gmail.com',
    maintainer='Stephan Rave',
    maintainer_email='stephan.rave@uni-muenster.de',
    license='BSD',
    url='http://github.com/pymumps/pymumps',
    packages=['mumps'],
    ext_modules=[
        Extension(
            'mumps._dmumps',
            sources=['mumps/_dmumps.pyx'],
            libraries=['dmumps', 'mumps_common', 'pord', 'openblas', 'mpiseq'],
        ),
        Extension(
            'mumps._smumps',
            sources=['mumps/_smumps.pyx'],
            libraries=['smumps', 'mumps_common', 'pord', 'openblas', 'mpiseq'],
        ),
        Extension(
            'mumps._zmumps',
            sources=['mumps/_zmumps.pyx'],
            libraries=['zmumps', 'mumps_common', 'pord', 'openblas', 'mpiseq'],
        ),
        Extension(
            'mumps._cmumps',
            sources=['mumps/_cmumps.pyx'],
            libraries=['cmumps', 'mumps_common', 'pord', 'openblas', 'mpiseq'],
        ),
    ],
    install_requires=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
