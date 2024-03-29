from setuptools import setup, find_packages

setup(name='deepinfo',
      version='0.0.3',
      description='Infer information theoretic quantities ( Mutual information, KL divergence, DJS divergence ) with Neural Networks',
      long_description='',
      author='Giulio Isacchini',
      author_email='giulioisac@gmail.com',
      license='GPLv3',
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Natural Language :: English',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.6',
            ],
      packages=['deepinfo'],
      install_requires=['numpy','tensorflow>=2.1.0','tqdm','scipy'],
      package_data = {},
      include_package_data=True,
      zip_safe=False)
