from setuptools import setup, find_packages

setup(
    name="fedgtd",
    version="2.0.0",
    description=(
        "Byzantine-Resilient Stochastic Games for Federated "
        "Multi-Cloud Intrusion Detection"
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "kagglehub>=0.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)
