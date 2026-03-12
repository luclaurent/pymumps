from pathlib import Path

_BASE_DIR = Path(__file__).parent / 'src'
_OUTPUT_DIR = _BASE_DIR /'tmp_pyx'

def clean_output_dir(output_dir=_OUTPUT_DIR):
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.is_file():
                f.unlink()
    else:
        output_dir.mkdir()

def generate_wrappers(type_num = ['s', 'd', 'c', 'z'], version="5.8.x", src_dir=_BASE_DIR, output_dir=_OUTPUT_DIR):
    

    if isinstance(type_num, str):
        type_num = [type_num]
        
    # adapted version
    new_version = ''
    if int(version[0]) < 5:
        raise ValueError('Version {} not supported, version should be 5.4.x, 5.5.x, 5.6.x, 5.7.x or 5.8.x'.format(version))
    elif int(version[0]) > 5:
        new_version = '5.7'
        print('Warning: version {} not recognized, using version {}'.format(version, new_version))
    #
    if int(version[2]) == 2:
        new_version = '5.1'
        print('Warning: version {} requested, using version {}'.format(version, new_version))
    elif int(version[2]) > 3 and int(version[2]) < 7:
        new_version = '5.3'
        print('Warning: version {} requested, using version {}'.format(version, new_version))
    elif int(version[2]) > 7 :
        new_version = '5.7'
        print('Warning: version {} requested, using version {}'.format(version, new_version))
    if new_version:
        version = new_version
    formatted_version  = version[:3] + '.x'
    
    
    # template
    _MUMPS_TEMPLATE = src_dir / ('_mumps_{}.tpl'.format(formatted_version))

    with open(_MUMPS_TEMPLATE, 'rt') as f:
        template = f.read()

    for x in type_num:
        with open(output_dir / f'_{x}mumps.pyx', 'wt') as f:
            f.write(template.format(x=x, X=x.upper()))
            
if __name__ == '__main__':
    generate_wrappers()
