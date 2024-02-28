"""Install SSSA object modelling package"""
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='sssa',
    version='1.0.0',
    author='Jonathan Gustafsson Frennert',
    description='SSSA Object Modelling Package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/J0HNN7G/sssa-object-modelling',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
    install_requires=[]
)