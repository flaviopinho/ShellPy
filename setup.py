from setuptools import setup, find_packages

# Lê o README.md para o long_description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Lê o requirements.txt para instalar dependências
with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="shellpy",
    version="0.2.0",
    author="Flávio Augusto Xavier Carneiro Pinho",
    description="Python library for shell analysis using the Ritz method and associated numerical techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flaviopinho/ShellPy",
    packages=find_packages(),
    python_requires="==3.12.*",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
