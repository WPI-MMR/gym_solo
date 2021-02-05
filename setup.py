from setuptools import find_packages, setup


setup(name='gym_solo',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym', 'pybullet'],
      extras_require={
        'test': ['parameterized', 'pyvirtualdisplay', 'xvfbwrapper']
      }
)
