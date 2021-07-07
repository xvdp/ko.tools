"""@xvdp
"""
import sys
import subprocess as sp
from setuptools import setup, find_packages

if sys.version_info[0] < 3:
    raise RuntimeError("Python 3+ required.")

def readme():
    with open('README.md') as _fo:
        return _fo.read()

def requirements():
    with open('requirements.txt') as _fo:
        return _fo.read().split()

def set_version(version):
    with open('kotools/version.py', 'w') as _fi:
        _fi.write("version='"+version+"'")
    return version



def setup_package():
    ''' setup '''

    metadata = dict(
        name='kotools',
        version=set_version(version='0.0.1'),
        description='common learning tools',
        url='https://github.com/xvdp/kotools',
        author='xvdp',
        license='MIT',
        install_requires=requirements(),
        packages=find_packages(),
        long_description=readme(),
        tests_require=["pytest"],
        include_package_data=True,
        zip_safe=False)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()