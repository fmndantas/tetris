from setuptools import setup

PREFIX = "__version__ = "

with open("tetris/__init__.py") as file:
    line = list(filter(lambda x: x.startswith(PREFIX), file.readlines()))[0].strip(PREFIX)
    __version__ = eval(line)

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tetris",
    version=__version__,
    author="Fernando Dantas",
    author_email="fernandodantas1996@gmail.com",
    license="MIT",
    description="Games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fmndantas/tetris",
    packages=["tetris"],
    scripts=["bin/tetris"],
    test_suite="tetris/tests/game_tests:GameTests",
    install_requires=["numpy", "keyboard", "opencv-python", "Pillow"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
        "Intended Audience :: Entertainment/Game",
    ],
)
