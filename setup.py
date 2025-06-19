from setuptools import setup, find_packages

setup(
    name="IVIMNET",
    version="0.1.1",
    author="Oliver Champion",
    description="Physics-informed deep learning IVIM model",
    url="https://github.com/oliverchampion/IVIMNET",
    license="GPL-3.0",
    packages=find_packages(where="."),
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "matplotlib",
        "scipy",
        "joblib",
    ],
    python_requires=">=3.7",
)