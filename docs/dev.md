# Setup Development

First, install uv to create a virtual env

Then, install `uv` and `tox`.

```shell
brew install uv -v
# creates a virtual env under .venv
uv tool install tox
```

Within that virtualenv:

```shell
$ uv sync
```

This will install development dependencies, followed by installing this package itself as ["editable"](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs).


## Run Tests

Tests can be invoked in two ways: `pytest` and `tox`.

### Run tests via `pytest`

This must be done within the virtualenv. Note that `pytest` will automatically pick up the config set in `tox.ini`. Comment it out if you want to skip coverage and/or ignore verbosity while iterating.

```sh
# for all tests
(env) $ pytest tests/

# for one module of tests
(env) $ pytest tests/test_main.py

# for one specific test
(env) $ pytest tests/test_main.py::test_return_state
```

`pytest` can either just run tests for you if you want to use Python's standard `unittest` library, or it can actually be used to define tests. People tend to find `pytest` a lot easier to use when writing tests because they're just simple `assert` statements. See `tests/main.py` for an example.

More info about pytest can be found [here](https://docs.pytest.org/en/latest/).

### Run tests via `tox`

`tox` must be done **outside** the virtualenv. This is because `tox` will create separate virtual environments for each test environment. A test environment could be based on python versions, or could be specific to documentation, or whatever else. See `tox.ini` as an example for three different test environments: running tests for Python 3.6, building docs to check for warnings & errors, and checking `MANIFEST.in` to assert the proper setup.

```sh
# run all environments
$ tox

# run a specific environment
$ tox -e docs
$ tox -e test
```

See [tox's documentation](https://tox.readthedocs.io/en/latest/) for more information.


## Managing dependencies

This repository uses `uv` to manage dependencies.
Requirements are specified in input files, e.g. [pyproject.toml](../pyproject.toml).

## Generate Documentation

Documentation can be generated and viewed via `mkdocs`:

```sh
# in the root of the repo
(env) $ mkdocs serve
```

## Release a New Version

To bump a version run [`bump-my-version`](https://github.com/callowayproject/bump-my-version) accordingly. For example:

```sh
# micro/patch version from 0.0.1 to 0.0.2
bumpver update --patch

# minor version from 0.0.2 to 0.1.0
bumpver update --minor

# major version from 0.1.0 to 1.0.0
bumpver update --major
```

Running `bumpversion` will create a commit and tag it automatically. Turn this off in `setup.cfg`.
.
