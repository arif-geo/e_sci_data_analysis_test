# note that these come from having run the pi & e approximations earlier
# define pi 
pi = 3.141592653589793

# define e
e = 2.718281828459045

# define the imaginary number i
i = 1.0j

# define the e**x function
def exp(x):
    """ Calculate e**x"""
    return e**x

def cos(x):
    """ Calculate cos(x) as real(e**(i*x))"""

    # calculate e**(i*x)
    z = exp(i*x)

    # return the real part for cos
    return z.real

def sin(x):
    """ Calculate cos(x) as real(e**(i*x))"""

    # calculate e**(i*x)
    z = exp(i*x)

    # return the imaginary part for sin
    return z.imag

def tan(x):
    """ Calculate tan(x) as sin(x)/cos(x)"""

    return sin(x)/cos(x)
