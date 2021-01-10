import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trapi",
    version="0.1.0",
    author="Shohei Goda",
    author_email="shohei.goda.12.1@gmail.com",
    description="You can use functions customized for machine learning cometitions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KenkenGoda/trapi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
