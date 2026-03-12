from pathlib import Path

_BASE_DIR = Path(__file__).parent / 'src'

def generate_wrappers(type_num = ['s', 'd', 'c', 'z'], version="5.8.x", src_dir=_BASE_DIR):

    if isinstance(type_num, str):
        type_num = [type_num]
        
    # adapted version
    formatted_version  = version[:3] + '.x'
    # template
    _MUMPS_TEMPLATE = src_dir / ('_mumps_{}.tpl'.format(formatted_version))

    with open(_MUMPS_TEMPLATE, 'rt') as f:
        template = f.read()

    for x in type_num:
        with open(src_dir / f'_{x}mumps.pyx', 'wt') as f:
            f.write(template.format(x=x, X=x.upper()))
            
if __name__ == '__main__':
    generate_wrappers()
