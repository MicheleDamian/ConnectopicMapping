from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='connectopic_mapping',

    version='0.1.0br1',

    description='Connectopic mapping',
    long_description=long_description,

    author='Michele Damian',
    author_email='michele.damian@gmail.com',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='neuroscience connectopic mapping research',

    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    py_modules=["connectopic_mapping"],

)
