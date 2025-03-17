from setuptools import setup, find_packages

setup(
    name='leanllm',
    version='0.0.1',
    author='simpx',
    author_email='simpxx@gmail.com',
    description='A lightweight framework for LLM inference',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/simpx/leanllm',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
