project('pymumps', 
'cython','c',
  version : '0.6.0', #run_command('src/_version.py', check: true).stdout().strip(),
#   license: 'BSD-3',
  meson_version: '>=1.5.0',
  default_options : ['warning_level=2', 'buildtype=release']
)
fs = import('fs')

# Python configuration
py = import('python').find_installation(pure: false)
datapy = py.get_path('data')
message(py.full_path())
message(py.get_install_dir())
message(datapy)

py_dep = py.dependency()

cc = meson.get_compiler('c')
# retrieve include dirs
r = run_command(cc.cmd_array() + ['-v', '-E', '-x', 'c', '-'], check: false)
sys_includes = []
read_inc = false
foreach line : r.stderr().split('\n')  
  if 'End of search list.' in line
    # Parse next lines until end marker
    read_inc = false
  endif
  if read_inc
    if '#include <...> search ends here.' in line
      read_inc = false
    else
      sys_includes += line.strip()
      message('System includes: ' + line.strip())
    endif
  endif
  if '#include <...> search starts here:' in line
    # Parse next lines until end marker
    read_inc = true
  endif
endforeach

# get mumps libraries
mumps_lib_d = cc.find_library('dmumps',  has_headers:['dmumps_c.h'], required: false)
mumps_lib_s = cc.find_library('smumps',  has_headers:['smumps_c.h'], required: false)
mumps_lib_z = cc.find_library('zmumps',  has_headers:['zmumps_c.h'], required: false)
mumps_lib_c = cc.find_library('cmumps',  has_headers:['cmumps_c.h'], required: false)
mumps_lib_common = cc.find_library('mumps_common', required: true)

if not mumps_lib_d.found() and not mumps_lib_s.found() and not mumps_lib_z.found() and not mumps_lib_c.found()
  error('Could not find any MUMPS library (dmumps, smumps, zmumps, cmumps). Please install MUMPS and ensure the libraries and headers are in the system path.')
endif

#locate define of MUMPS version
tmp_header = ''
if mumps_lib_z.found()
  tmp_header = 'zmumps_c.h'
elif mumps_lib_s.found()
  tmp_header = 'smumps_c.h'
elif mumps_lib_d.found()
  tmp_header = 'dmumps_c.h'
elif mumps_lib_c.found()
  tmp_header = 'cmumps_c.h'
endif
# search for define along include dirs
foreach incdir : sys_includes
  message('Searching for MUMPS_VERSION in: ' + incdir)
  VAR = ''
  if fs.is_dir(incdir)
    if cc.has_header(tmp_header, include_directories: incdir)
      message ('Found header: ' + tmp_header + ' in ' + incdir)
      header_path = join_paths(incdir, tmp_header)
      if fs.exists(header_path)
        message('Header path: ' + header_path)
        VAR = cc.get_define('MUMPS_VERSION',
          include_directories: [incdir],
          prefix: '#include "' + header_path + '"')
      endif
    endif    
  endif
  if VAR != ''
    message('Found MUMPS_VERSION: ' + VAR + ' in ' + incdir)
    FOUND_MUMPS_VERSION = VAR
  endif
endforeach

if FOUND_MUMPS_VERSION == ''
  error('Could not find MUMPS_VERSION in any of the system include directories.')
endif

# get MUMPS version from automatic chosen includes
MUMPS_VERSION = cc.get_define('MUMPS_VERSION',
          prefix: '#include "' + tmp_header + '"')
message('Found MUMPS_VERSION: ' + VAR + ' in ' + incdir)

message(mumps_lib_z.name())
message(mumps_lib_z.type_name())
message(VAR)
# message(mumps_lib_z.fullpath())
# message(mumps_lib_z.partial_dependency(includes:true).found())


# Generate Cython wrappers
python3 = import('python')
py = python3.find_installation(pure: false)
run_command(py,
['-c',
'import generate_wrappers as gen; gen.clean_output_dir()'], 
check: true)

if mumps_lib_d.found()
run_command(py,
['-c',
'import generate_wrappers as gen; gen.generate_wrappers(type_num = "d", version=' + MUMPS_VERSION + ')'], 
check: true)
endif

if mumps_lib_s.found()
run_command(py,
['-c',
'import generate_wrappers as gen; gen.generate_wrappers(type_num = "s", version=' + MUMPS_VERSION + ')'], 
check: true)
endif

if mumps_lib_z.found()
run_command(py,
['-c',
'import generate_wrappers as gen; gen.generate_wrappers(type_num = "z", version=' + MUMPS_VERSION + ')'], 
check: true)
endif

if mumps_lib_c.found()
run_command(py,
['-c',
'import generate_wrappers as gen; gen.generate_wrappers(type_num = "c", version=' + MUMPS_VERSION + ')'], 
check: true)
endif


# 
if mumps_lib_d.found()
  message('Add dmumps Python module')
  cython_source_d = files('src/tmp_pyx/_dmumps.pyx')
  py.extension_module(
  '_dmumps', 
  cython_source_d,
  dependencies : [py_dep, mumps_lib_d, mumps_lib_common],
  install: true,
  subdir: 'mumps',  # Folder relative to site-packages to install to
  include_directories: ['src']
  )
endif

if mumps_lib_s.found()
  message('Add smumps Python module')
  cython_source_s = files('src/tmp_pyx/_smumps.pyx')
  py.extension_module(
  '_smumps', 
  cython_source_s,
  dependencies : [py_dep, mumps_lib_s, mumps_lib_common],
  install: true,
  subdir: 'mumps',  # Folder relative to site-packages to install to
  include_directories: ['src']
  )
endif

if mumps_lib_z.found()
  message('Add zmumps Python module')
  cython_source_z = files('src/tmp_pyx/_zmumps.pyx')
  py.extension_module(
  '_zmumps', 
  cython_source_z,
  dependencies : [py_dep, mumps_lib_z, mumps_lib_common],
  install: true,
  subdir: 'mumps',  # Folder relative to site-packages to install to
  include_directories: ['src']
  )
endif

if mumps_lib_c.found()
  message('Add cmumps Python module')
  cython_source_c = files('src/tmp_pyx/_cmumps.pyx')
  py.extension_module(
  '_cmumps', 
  cython_source_c,
  dependencies : [py_dep, mumps_lib_c, mumps_lib_common],
  install: true,
  subdir: 'mumps',  # Folder relative to site-packages to install to
  include_directories: ['src']
  )
endif

message('Processing done: ')

# Declare the sources for Python
py.install_sources([
  'src/__init__.py',
  'src/_version.py',
  'src/tests/test_mumps.py'
  ],
  pure: false,
  preserve_path: false,
  subdir: 'mumps'
  )


