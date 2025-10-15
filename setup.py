import setuptools
from setuptools.command.develop import develop
import os
from huggingface_hub import snapshot_download

class DownloadCheckpoints(develop):
    def run(self):
        super().run()
        # pusht
        pusht_path = os.path.join(os.getcwd(), "llfbench/envs/pusht/oracles/")
        os.makedirs(pusht_path, exist_ok=True)
        print("Downloading pusht-keypoint-v0 checkpoints from Hugging Face ...")
        try:
            snapshot_download(repo_id="LLM-BC/pushT-checkpoints", repo_type="model", local_dir=pusht_path)
            print(f"Checkpoint downloaded successfully to {pusht_path}")
        except Exception as e:
            print(f"Failed to download pusht checkpoint: {e}")

        # parking
        parking_path = os.path.join(os.getcwd(), "llfbench/envs/highway/oracles/")
        os.makedirs(parking_path, exist_ok=True)
        print("Downloading parking-v0 checkpoints from Hugging Face ...")
        try:
            snapshot_download(repo_id="LLM-BC/parking-checkpoints", repo_type="model", local_dir=parking_path)
            print(f"Checkpoint downloaded successfully to {parking_path}")
        except Exception as e:
            print(f"Failed to download parking checkpoint: {e}")

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
        "gym==0.23.1",
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
        'requests==2.32.0',
        'omegaconf',
        'huggingface-hub==0.26.2',
        # d4rl
        'd4rl',
        'cython<3',
    ],
    extras_require={
        'metaworld': ['metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@c822f28#egg=metaworld'],
        'alfworld': [ 'alfworld>=0.3.0' ],
        'maniskill': [ 'mani_skill==3.0.0b20' ],
        'blockpushing': [ 
            'tf-agents==0.19.0', 
        ],
        'pusht': [
            'pymunk==6.2.1',
            'scikit-image==0.19.3',
            'shapely==1.8.4',
            'cffi==1.17.1'
        ],
        'highway': [
            'stable-baselines3==2.2.1'
        ]
    },
    cmdclass={'develop': DownloadCheckpoints}
)