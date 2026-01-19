from setuptools import setup, find_packages

# setup(
#     name="Ner_Pipeline",
#     version="0.1.1",
#     packages=find_packages(),  # auto‐discovers packages
# )

setup(
    name="ner_pipeline",
    version="0.1.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
)