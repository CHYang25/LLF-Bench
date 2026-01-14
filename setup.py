import os
import io
import setuptools
from setuptools.command.develop import develop

class DownloadCheckpoints(develop):
    user_options = develop.user_options + [
        ('skip-checkpoints', None, 'Skip downloading checkpoints'),
    ]

    boolean_options = develop.boolean_options + ['skip-checkpoints']

    def initialize_options(self):
        super().initialize_options()
        # Allow skipping via flag or env var
        self.skip_checkpoints = bool(os.environ.get('LLFBENCH_SKIP_CHECKPOINTS'))

    def run(self):
        super().run()

        if self.skip_checkpoints:
            print("[llfbench] Skipping checkpoint download (flag/env).")
            return

        # Import here to avoid build-time import during metadata/isolated build
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            print(
                "[llfbench] Could not import huggingface_hub during post-develop step.\n"
                f"Reason: {e}\n"
                "Install it first or re-run:\n"
                "  pip install 'huggingface-hub>=0.16'\n"
                "Then manually fetch checkpoints with:\n"
                "  python -m huggingface_hub download LLM-BC/pushT-checkpoints --repo-type model -d llfbench/envs/pusht/oracles/\n"
                "  python -m huggingface_hub download LLM-BC/parking-checkpoints --repo-type model -d llfbench/envs/highway/oracles/\n"
            )
            return

        # pusht
        pusht_path = os.path.join(os.getcwd(), "llfbench", "envs", "pusht", "oracles")
        os.makedirs(pusht_path, exist_ok=True)
        print("[llfbench] Downloading pusht-keypoint-v0 checkpoints from Hugging Face ...")
        try:
            snapshot_download(
                repo_id="LLM-BC/pushT-checkpoints",
                repo_type="model",
                local_dir=pusht_path,
                local_dir_use_symlinks=False,
            )
            print(f"[llfbench] pusht checkpoints downloaded to {pusht_path}")
        except Exception as e:
            print(f"[llfbench] Failed to download pusht checkpoint: {e}")

        # parking
        parking_path = os.path.join(os.getcwd(), "llfbench", "envs", "highway", "oracles")
        os.makedirs(parking_path, exist_ok=True)
        print("[llfbench] Downloading parking-v0 checkpoints from Hugging Face ...")
        try:
            snapshot_download(
                repo_id="LLM-BC/parking-checkpoints",
                repo_type="model",
                local_dir=parking_path,
                local_dir_use_symlinks=False,
            )
            print(f"[llfbench] parking checkpoints downloaded to {parking_path}")
        except Exception as e:
            print(f"[llfbench] Failed to download parking checkpoint: {e}")

def _read_readme():
    try:
        with io.open('README.md', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""

setuptools.setup(
    name='llfbench',
    version='0.1.0',
    author='LLF-Bench Team',
    author_email='chinganc@microsoft.com',
    url='https://github.com/microsoft/LLF-Bench',
    license='MIT',
    description='A Gym environment for Learning from Language Feedback (LLF).',
    long_description=_read_readme(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(include=["llfbench*"], exclude=["tests*"]),
    include_package_data=True,
    install_requires=[
        "numpy<1.24.0",
        "tqdm",
        "gymnasium==0.29.1",
        "gym==0.23.1",
        "parse==1.19.1",
        "openai==2.8.1",
        "pyautogen==0.1",
        "gym-bandits@git+https://github.com/JKCooper2/gym-bandits#egg=gym-bandits",
        "cmudict==1.0.13",
        "syllables==1.0.9",
        "jax",
        "jaxlib",
        "highway-env",
        "requests==2.32.0",
        "omegaconf",
        "huggingface-hub==0.26.2",
        "d4rl",
        "cython<3",
        "mujoco==2.3.7",
    ],
    extras_require={
        'metaworld': ['metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@c822f28#egg=metaworld'],
        'alfworld': [ 'alfworld>=0.3.0' ],
        'maniskill': [ 'mani_skill==3.0.0b20' ],
        'blockpushing': [ 'tf-agents==0.19.0' ],
        'pusht': [
            'pymunk==6.2.1',
            'scikit-image==0.19.3',
            'shapely==1.8.4',
            'cffi==1.17.1'
        ],
        'highway': [ 'stable-baselines3==2.2.1' , "mujoco==2.3.7" ]
    },
    # Keeps your current behavior for editable installs,
    # but now it's safe if huggingface_hub isn't available during build.
    cmdclass={'develop': DownloadCheckpoints},
)
