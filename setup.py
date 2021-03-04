from subprocess import call
from sys import executable
from setuptools import setup, find_packages
from setuptools.command.install import install


class Install(install):
    """Installation procedure.

    Parameters
    ----------
    install : setuptools.command.install.install
    """

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        install.run(self)
        call([
            executable,
            "-m",
            "nltk.downloader",
            "stopwords",
            "punkt"
        ], shell=False)


setup(
    name="ragged_text",
    version="0.1.4",
    description="Components for text classification with Tensorflow",
    author="Dave Hollander",
    author_url="https://github.com/brainsqueeze",
    url="https://github.com/brainsqueeze/ragged_text_classification",
    license="MIT",
    install_requires=[
        "nltk",
        "numpy",
        "tensorflow>=2.4.1",
        "text2vec @ git+https://github.com/brainsqueeze/text2vec@master#egg=text2vec"
    ],
    packages=find_packages(exclude=["bin"]),
    cmdclass={"install": Install}
)
