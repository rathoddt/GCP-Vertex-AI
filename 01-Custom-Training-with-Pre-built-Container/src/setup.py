from setuptools import find_packages, setup

setup(
    name='sentiment_analysis',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    description='My Sentiment Analysis Tool',
    install_requires=[
        "pandas",
        "scikit-learn",
        "google-cloud-storage",
    ],
)
