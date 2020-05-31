import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Stock_Analysis-Liam_Morrow",
    version="0.0.1",
    author="Liam Morrow",
    author_email="morrow.liam@gmail.com",
    description="Stock Analysis to evaluate my personal accounts.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MorrowLiam/stock_analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
