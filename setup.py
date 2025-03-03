import setuptools

setuptools.setup(
    name='llfbench',
    version='0.1.0',
    author='LLF-Bench Team',
    author_email='chinganc@microsoft.com',
    packages=setuptools.find_packages(include=["llfbench*"], exclude=["tests*"]),
    url='https://github.com/microsoft/LLF-Bench',
    license='MIT LICENSE',
    description='A Gym environment for Learning from Language Feedback (LLF).',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy<1.24.0",
        "tqdm",
        "gymnasium==0.29.1",
        "parse==1.19.1",
        "openai==0.28",
        "pyautogen==0.1",
        # "Cython==0.29.36",
        "gym-bandits@git+https://github.com/JKCooper2/gym-bandits#egg=gym-bandits",
        # poem
        "cmudict==1.0.13",
        "syllables==1.0.9",
        # optimization
        "jax",
        "jaxlib",
        # highway
        "highway-env",
        # reco
        'requests==2.32.0'
    ],
    extras_require={
        'metaworld': ['metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@c822f28#egg=metaworld'],
        'alfworld': [ 'alfworld>=0.3.0' ],
        'maniskill': [ 'mani_skill==3.0.0b12' ],
        'blockpushing': [ 
            'pybullet==3.2.6', 
            'tf-agents==0.19.0', 
            'typing-extensions==4.12.2',
            'pygame==2.6.1',
            'gym==0.26.2',
            'tensorflow==2.15.1'
        ]
    }
)
