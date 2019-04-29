from setuptools import setup, find_packages
import os

def read(fname):
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='arxiv_public_data',
      version='0.1.1',
      description='Download and process publicly available ArXiv data',
      long_description=read('README.md'),
      keywords='data-mining citation-networks arxiv',
      url='',
      author="Matt Bierbaum, Colin B. Clement, Kevin P. O'Keeffe, and Alex A. Alemi",
      author_email='colin.clement@gmail.com',
      license='MIT',
      packages=find_packages(),  # exclude=['test*']),
      install_requires=[''],
      python_requires='>=3.6, <4',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Information Analysis'
      ]
)
