from setuptools import setup


def text_of_readme_md_file():
    try:
        with open('README.md') as f:
            return f.read()
    except:
        return ""


setup(
    long_description=text_of_readme_md_file(),
    long_description_content_type="text/markdown"
)  # Note: Everything should be in the local setup.cfg

#
#
# import os
# from configparser import ConfigParser
# from setuptools import find_packages
#
# version = None
#
# def my_setup(print_params=True, **setup_kwargs):
#     from setuptools import setup
#     if print_params:
#         import json
#         print("Setup params -------------------------------------------------------")
#         print(json.dumps(setup_kwargs, indent=2))
#         print("--------------------------------------------------------------------")
#     setup(**setup_kwargs)
#
#
# root_dir = os.path.dirname(__file__)
#
# config_file = os.path.join(root_dir, 'setup.cfg')
#
# c = ConfigParser()
# c.read_file(open(config_file, 'r'))
# name = c['metadata']['name']
# root_url = c['metadata']['root_url']
#
# more_setup_kwargs = dict(
#     c['metadata'],
#     install_requires=[
#         'matplotlib',
#         'py2store',
#         'slang',
#         'soundfile',
#         'scikit-learn',
#         'pandas',
#         'importlib_resources'
#     ],
#     keywords=['data', 'data access', 'data preperation', 'machine learning', 'artificial intelligence'],
# )
#
# # import os
# # name = os.path.split(os.path.dirname(__file__))[-1]
#
# # version = '0.0.2'  # edit if you want to specify the version here (should be a string)
# if version is None:
#     try:
#         from pip_packaging import next_version_for_package
#
#         version = next_version_for_package(name)  # when you want to make a new package
#     except Exception as e:
#         print(f"Got an error trying to get the new version of {name} so will try to get the version from setup.cfg...")
#         print(f"{e}")
#         version = c['metadata'].get('version', None)
#         if version is None:
#             raise ValueError(f"Couldn't fetch the next version from PyPi (no API token?), "
#                              f"nor did I find a version in setup.cfg (metadata section).")
#
#
# def readme():
#     try:
#         with open('README.md') as f:
#             return f.read()
#     except:
#         return ""
#
#
# ujoin = lambda *args: '/'.join(args)
#
# if root_url.endswith('/'):
#     root_url = root_url[:-1]
#
# dflt_kwargs = dict(
#     name=f"{name}",
#     version=f'{version}',
#     url=f"{root_url}/{name}",
#     packages=find_packages(),
#     include_package_data=True,
#     platforms='any',
#     long_description=readme(),
#     long_description_content_type="text/markdown",
# )
#
# setup_kwargs = dict(dflt_kwargs, **more_setup_kwargs)
#
# ##########################################################################################
# # Diagnose setup_kwargs
# _, containing_folder_name = os.path.split(os.path.dirname(__file__))
# if setup_kwargs['name'] != containing_folder_name:
#     print(f"!!!! containing_folder_name={containing_folder_name} but setup name is {setup_kwargs['name']}")
#
# ##########################################################################################
# # Okay... set it up alright!
# my_setup(**setup_kwargs)
