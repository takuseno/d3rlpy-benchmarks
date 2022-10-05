from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="d3rlpy-benchmarks",
        version="0.1.0",
        description="A large-scale benchmark for deep offline reinforcement learning",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/takuseno/d3rlpy-benchmarks",
        author="Takuma Seno",
        author_email="takuma.seno@gmail.com",
        license="MIT License",
        install_requires=["numpy", "matplotlib", "seaborn", "scipy", "rliable"],
        packages=find_packages(exclude=["tests*"]),
        python_requires=">=3.7.0",
    )
