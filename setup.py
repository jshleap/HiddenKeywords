import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ADvantage",
    version="0.0.1",
    author="Jose Sergio Hleap",
    author_email="jshleap@gmail.com",
    description="This package crawls a landing page, get relevant keywords and"
                " optimize the best choice of keywords for a SEO campaign",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jshleap/HiddenKeywords.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3 licence",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)