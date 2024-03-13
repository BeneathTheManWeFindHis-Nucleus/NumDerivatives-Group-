from richardson import richardson
from dual import dual
from chebyshev import chebyshev

def derivative(f, x, method):
    if method == 'Richardson':
        return richardson(f, x)
    elif method == 'Dual':
        return dual(f, x)
    elif method == 'Chebyshev':
        return chebyshev(f, a, b)
    else:
        raise ValueError('Invalid method')
    
""" For Chebyshev polynomials, the domain has to be defined as a transformation of 
the roots of the polynomials from [-1,1] to endpoints [a,b]. Could we define our
interface to accept the endpoints as arguments (f, a, b) to accomodate this? """