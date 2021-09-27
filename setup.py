"""@xvdp
"""
import setuptools
from setuptools import setup, find_packages
import os.path as osp

def readme():
    with open('README.md') as _fo:
        return _fo.read()

def requirements():
    req = osp.join(osp.split(__file__)[0], 'requirements.txt')
    with open(req) as _fo:
        return _fo.read().split()

# def set_version(version):
#     with open('kotools/version.py', 'w') as _fi:
#         _fi.write("version='"+version+"'")
#     return version

# def setup_package():
#     ''' setup '''

#     metadata = dict(
#         name='kotools',
#         version=set_version(version='0.0.3'),
#         description='common learning tools',
#         url='https://github.com/xvdp/kotools',
#         author='xvdp',
#         license='MIT',
#         install_requires=requirements(),
#         packages=find_packages(),
#         long_description=readme(),
#         tests_require=["pytest"],
#         include_package_data=True)

#     setup(**metadata)

# if __name__ == '__main__':
#     setup_package()

setuptools.setup(
    name='kotools',
    version="0.0.4",
    author="xvdp",
    license='MIT',
    install_requires=requirements(),
    description='kotools, spome common learning tools',
    long_description=readme(),
    url='https://github.com/xvdp/kotools',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    tests_require=["pytest"],
    python_requires='>=3.6',
)
