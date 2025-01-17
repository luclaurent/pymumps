project('pymumps', 
'cython',
  version : '0.6.0', #run_command('src/_version.py', check: true).stdout().strip(),
#   license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : ['warning_level=2', 'buildtype=release']
)

# Python configuration
py = import('python').find_installation(pure: false)
message(py.full_path())
message(py.get_install_dir())

py_dep = py.dependency()

# Define the source files
cython_sources = files('src/_zmumps.pyx')

# Assumes that f2py was run before in "name" module
py.extension_module(
'_zmumps', 
cython_sources,
dependencies : [py_dep],
install: true,
# link_with: obj_dep,
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


