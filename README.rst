pdexplore
#########
|PyPI-Status| |PyPI-Versions| |Build-Status| |Codecov| |LICENCE|

Basic data exploration for pandas DataFrames.

.. code-block:: python

  >>> df = pd.DataFrame(
          data=[[4, 165, 'USA'], [2, 180, 'UK'], [2, 170, 'Greece']],
          index=['Dana', 'Jane', 'Nick'],
          columns=['Medals', 'Height', 'Born']
      )
  >>> import pdexplore as pde
  >>> pde.explore(df)

.. contents::

.. section-numbering::

Installation
============

Install ``pdexplore`` with:

.. code-block:: bash

  pip install pdexplore


Features
========

* A simple interface.
.. * Fully tested.
* Compatible with Python 3.5+.
* Pure Python.
* Colored output using `colored <https://gitlab.com/dslackw/colored>`_.


Basic Use
=========

To let pdexplore go through each column of a dataframe and explore it, simple call the ``explore`` function:

.. code-block:: python

  >>> df = pd.DataFrame(
          data=[[4, 165, 'USA'], [2, 180, 'UK'], [2, 170, 'Greece']],
          index=['Dana', 'Jane', 'Nick'],
          columns=['Medals', 'Height', 'Born']
      )
  >>> import pdexplore as pde
  >>> pde.explore(df)


Properties explored
===================

Column-wise explorations
------------------------

General
~~~~~~~

* Number of unique values.
* Number and rate of missing values.
* A nice value counts plot for most frequent values.

Numeric
~~~~~~~

* Basic statistics: Min, max, mean, standard deviation, median and median absolute deviation (MAD).
* Skewness measure and a skewness test for normal-like skewness.
* Normality tests:

  * Shapiro-Wilk test.
  * D'Agostino-Pearson K^2 test.
* Checking for suspicious values:

  * Min and max values for signed and unsigned integers for 8, 16, 32, 64 and 128-bit integers; checks for both occurence in the data, and for data length (i.e. the number of records).
  * All 9 numbers (often used as sentinals).


Contributing
============

Package author and current maintainer is Shay Palachy (shay.palachy@gmail.com); You are more than welcome to approach him for help. Contributions are very welcomed, especially since this package is very much in its infancy and many other pipeline stages can be added.

Installing for development
--------------------------

Clone:

.. code-block:: bash

  git clone git@github.com:shaypal5/pdexplore.git


Install in development mode with test dependencies:

.. code-block:: bash

  cd pdexplore
  pip install -e ".[test]"


Running the tests
-----------------

To run the tests, use:

.. code-block:: bash

  python -m pytest --cov=pdexplore


Adding documentation
--------------------

This project is documented using the `numpy docstring conventions`_, which were chosen as they are perhaps the most widely-spread conventions that are both supported by common tools such as Sphinx and result in human-readable docstrings (in my personal opinion, of course). When documenting code you add to this project, please follow `these conventions`_.

.. _`numpy docstring conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`these conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

Additionally, if you update this ``README.rst`` file,  use ``python setup.py checkdocs`` to validate it compiles.


Credits
=======
Created by Shay Palachy  (shay.palachy@gmail.com).

.. alternative:
.. https://badge.fury.io/py/yellowbrick.svg

.. |PyPI-Status| image:: https://img.shields.io/pypi/v/pdexplore.svg
  :target: https://pypi.org/project/pdexplore

.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/pdexplore.svg
   :target: https://pypi.org/project/pdexplore

.. |Build-Status| image:: https://travis-ci.org/shaypal5/pdexplore.svg?branch=master
  :target: https://travis-ci.org/shaypal5/pdexplore

.. |LICENCE| image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://pypi.python.org/pypi/pdexplore
  
.. .. |LICENCE| image:: https://github.com/shaypal5/pdexplore/blob/master/mit_license_badge.svg
  :target: https://pypi.python.org/pypi/pdexplore
  
.. https://img.shields.io/pypi/l/pdexplore.svg

.. |Codecov| image:: https://codecov.io/github/shaypal5/pdexplore/coverage.svg?branch=master
   :target: https://codecov.io/github/shaypal5/pdexplore?branch=master
