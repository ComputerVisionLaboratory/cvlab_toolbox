from distutils.core import setup

setup(
    name = 'cvt',
    version='0.0',
    packages=[
        'cvt',
        'cvt.evaluation',
    ],
    install_requires=['numpy'],
)
