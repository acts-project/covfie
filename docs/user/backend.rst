Vector field backends
=====================

In covfie, the majority of the behaviour of your vector field is determined by
the field :dfn:`backend`. In covfie, backends are divided into two categories:
initial backends and composite backends.

Initial backends
----------------

An :dfn:`initial backend` is a backend that is not composed of transformers.
Initial backends represent basic functionality that cannot -- for reasons of
practicality or performance -- be decomposed into smaller components. Initial
backends can be used directly, but often lack the necessary functionality to
model real-world vector fields.

Memory backends
~~~~~~~~~~~~~~~

Constant backends
~~~~~~~~~~~~~~~~~

Analytical backends
~~~~~~~~~~~~~~~~~~~

Transformers
------------

Backend transformers are useless on their own, but can be used to add
additional functionality to existing backends. Backend transformers form the
core of covfie's compositional nature.

Storage order backends
~~~~~~~~~~~~~~~~~~~~~~

Clamping backends
~~~~~~~~~~~~~~~~~

Interpolation backends
~~~~~~~~~~~~~~~~~~~~~~

Geometric backends
~~~~~~~~~~~~~~~~~~

Practical limitations
---------------------
