from setuptools import setup, find_packages

setup(
    name="PyDocSmith",
    version="0.1.1",
    description="It parses and composes docstrings in Google, Numpydoc, ReST and Epydoc.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Suman Saurabh",
    author_email="ss.sumansaurabh92@gmail.com",
    url="https://github.com/SingularityX-ai/PyDocSmith",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)