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
    name='PyMUMPS5',
    version='0.3.2',
    description='Python bindings for MUMPS5, a parallel sparse direct solver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Bradley M. Froehle',
    author_email='brad.froehle@gmail.com',
    maintainer='Stephan Rave',
    maintainer_email='stephan.rave@uni-muenster.de',
    license='BSD',
    url='http://github.com/pymumps/pymumps',
    packages=['mumps5'],
    ext_modules=[
        Extension(
            'mumps5._dmumps5',
            sources=['mumps5/_dmumps5.pyx'],
            libraries=['dmumps5', 'mumps_common5', 'pord5', 'openblas', 'mpiseq5'],
        ),
        Extension(
            'mumps5._smumps5',
            sources=['mumps5/_smumps5.pyx'],
            libraries=['smumps5', 'mumps_common5', 'pord5', 'openblas', 'mpiseq5'],
        ),
        Extension(
            'mumps5._zmumps5',
            sources=['mumps5/_zmumps5.pyx'],
            libraries=['zmumps5', 'mumps_common5', 'pord5', 'openblas', 'mpiseq5'],
        ),
        Extension(
            'mumps5._cmumps5',
            sources=['mumps5/_cmumps5.pyx'],
            libraries=['cmumps5', 'mumps_common5', 'pord5', 'openblas', 'mpiseq5'],
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
