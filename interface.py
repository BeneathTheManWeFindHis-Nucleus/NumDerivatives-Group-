from richardson import richardson
from dual import dual
from chebyshev import chebyshev

def derivative(f, x, method):
    if method == 'Richardson':
        return richardson(f, x)
    elif method == 'Dual':
        return dual(f, x)
    elif method == 'Chebyshev':
        return chebyshev(f, x)
    else:
        raise ValueError('Invalid method')
    
