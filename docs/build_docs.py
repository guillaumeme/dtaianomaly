
import os
import subprocess
import toml
import shutil


# Load all the versions to build the documentation from
with open('versions.toml', 'r') as versions_toml:
    versions = toml.load(versions_toml)['versions']


# Build the documentation for the specific version from given tag
def build_doc(version_to_build, tag):
    os.environ["current_version"] = version_to_build
    subprocess.run(f'git fetch origin {tag}', shell=True)
    subprocess.run(f'git tag', shell=True)
    subprocess.run(f'git checkout {tag}', shell=True)
    subprocess.run("git checkout main -- conf.py", shell=True)  # Which conf do we want? Preferably from the current tag I guess
    subprocess.run("make html", shell=True)



def move_to_tmp(version_dir):
    os.makedirs(f'./tmp/{version_dir}', exist_ok=True)
    for file_name in os.listdir('./_build/html/'):
        shutil.move(f'./_build/html/{file_name}', f'./tmp/{version_dir}/{file_name}')


def move_tmp_to_html():
    for file_name in os.listdir('./tmp/'):
        shutil.move(f'./tmp/{file_name}', f'./_build/html/{file_name}')


# Make the documentation for each version, and move the directory accordingly
for version in versions:
    build_doc(version, version)
    move_to_tmp(version.replace('.', '_'))

# Manually build the latest version from the main branch
build_doc("latest", "main")

move_tmp_to_html()
os.rmdir('./tmp')
