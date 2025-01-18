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
mumps_lib = []
mumps_lib += cc.find_library('dmumps', required: false)
mumps_lib += cc.find_library('mumps_common', required: false)
# message(zmumps.name())
# message(zmumps.type_name())

# Define the source files
cython_sources = files('src/_dmumps.pyx')
# cython_sources = files('src/_zmumps.pyx')
# cython_sources = files('src/_zmumps.pyx')
# cython_sources = files('src/_zmumps.pyx')

# Define the dependencies
# zmumps = dependency('zmumps', required: false)
# message(zmumps)
# if not zmumps.found()
# message('ZMUMPS not found')
# zmumpslib = declare_dependency(
# link_args : ['-L/usr/local/lib', '-lzmumps' ],
# include_directories : include_directories('/usr/local/include')
# )
# endif

# Assumes that f2py was run before in "name" module
py.extension_module(
'_dmumps', 
cython_sources,
dependencies : [py_dep, mumps_lib],
# c_args:['-lzmumps'],
install: true,
# link_with: zmumps,
# dependencies: [zmumpslib],
# link_language: 'fortran',
# native: true,
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


