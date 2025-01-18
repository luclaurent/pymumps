project('pymumps', 
'cython','c',
  version : '0.6.0', #run_command('src/_version.py', check: true).stdout().strip(),
#   license: 'BSD-3',
#   meson_version: '>=0.64.0',
  default_options : ['warning_level=2', 'buildtype=release']
)

# Python configuration
py = import('python').find_installation(pure: false)
datapy = py.get_path('data')
message(py.full_path())
message(py.get_install_dir())
message(datapy)

py_dep = py.dependency()

cc = meson.get_compiler('c')
mumps_lib_d = cc.find_library('dmumps', required: false)
mumps_lib_s = cc.find_library('smumps', required: false)
mumps_lib_z = cc.find_library('zmumps', required: false)
mumps_lib_x = cc.find_library('cmumps', required: false)
mumps_lib_common = cc.find_library('mumps_common', required: false)
# message(zmumps.name())
# message(zmumps.type_name())

# Define the source files
cython_source_d = files('src/_dmumps.pyx')
cython_source_s = files('src/_smumps.pyx')
cython_source_z = files('src/_zmumps.pyx')
cython_source_c = files('src/_cmumps.pyx')

# 
py.extension_module(
'_dmumps', 
cython_sources,
dependencies : [py_dep, mumps_lib_d, mumps_lib_common],
install: true,
subdir: 'pymumps',  # Folder relative to site-packages to install to
include_directories: ['src']
)

py.extension_module(
'_smumps', 
cython_sources,
dependencies : [py_dep, mumps_lib_s, mumps_lib_common],
install: true,
subdir: 'pymumps',  # Folder relative to site-packages to install to
include_directories: ['src']
)

py.extension_module(
'_zmumps', 
cython_sources,
dependencies : [py_dep, mumps_lib_z, mumps_lib_common],
install: true,
subdir: 'pymumps',  # Folder relative to site-packages to install to
include_directories: ['src']
)

py.extension_module(
'_cmumps', 
cython_sources,
dependencies : [py_dep, mumps_lib_c, mumps_lib_common],
install: true,
subdir: 'pymumps',  # Folder relative to site-packages to install to
include_directories: ['src']
)

message('Processing done: ')

# Declare the sources for Python
py.install_sources([
  'src/__init__.py',
  'src/_version.py',
  'src/tests/test_mumps.py'
  ],
  pure: false,
  preserve_path: false,
  subdir: 'pymumps'
  )


