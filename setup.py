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
    version='0.5.0',
    description='Python bindings for MUMPS v 5.4.1, a parallel sparse direct solver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Bradley M. Froehle',
    author_email='brad.froehle@gmail.com',
    maintainer='Olivier De Smet',
    maintainer_email='olivier.desmet@lecnam.net',
    license='BSD',
    url='http://github.com/luclaurent/pymumps',
    packages=['mumps'],
    ext_modules=[
        Extension(
            'mumps._dmumps',
            sources=['mumps/_dmumps.pyx'],
            # include_dirs=['/usr/local/include'],
            # library_dirs=['/usr/local/lib'],
            libraries=['dmumps', 'mumps_common'] #, 'esmumps', 'pord', 'metis', 'scotch', 'scotchmetis', 'scotcherr', 'scotcherrexit', 'openblas', 'mpiseq', 'z'],
        ),
        Extension(
            'mumps._smumps',
            sources=['mumps/_smumps.pyx'],
            # include_dirs=['/usr/local/include'],
            # library_dirs=['/usr/local/lib'],
            libraries=['smumps', 'mumps_common'] #, 'esmumps', 'pord', 'metis', 'scotch', 'scotchmetis', 'scotcherr', 'scotcherrexit', 'openblas', 'mpiseq', 'z'],
        ),
        Extension(
            'mumps._zmumps',
            sources=['mumps/_zmumps.pyx'],
            # include_dirs=['/usr/local/include'],
            # library_dirs=['/usr/local/lib'],
            libraries=['zmumps', 'mumps_common'] #, 'esmumps', 'pord', 'metis', 'scotch', 'scotchmetis', 'scotcherr', 'scotcherrexit', 'openblas', 'mpiseq', 'z'],
        ),
        Extension(
            'mumps._cmumps',
            sources=['mumps/_cmumps.pyx'],
            # include_dirs=['/usr/local/include'],
            # library_dirs=['/usr/local/lib'],
            libraries=['cmumps', 'mumps_common'] #, 'esmumps', 'pord', 'metis', 'scotch', 'scotchmetis', 'scotcherr', 'scotcherrexit', 'openblas', 'mpiseq', 'z'],
        ),
    ],
    install_requires=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
