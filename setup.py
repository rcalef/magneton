from setuptools import setup

setup(
    name='magneton',
    version='0.1',
    description='Substructure-aware protein representation learning',
    url='http://github.com/rcalef/magneton',
    author='Robert Calef, Arthur Liang',
    author_email='rcalef@mit.edu, artliang@mit.edu',
    license='MIT',
    packages=['magneton'],
    install_requires=[
      "fire",
      "pysam",
    ],
    zip_safe=False,
)
