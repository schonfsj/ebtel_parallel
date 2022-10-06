# Contains statistical functions to calculate power laws
# with difined minima, maxima, and mean/median.
import sys
import warnings
warnings.filterwarnings('ignore','invalid value encountered in float_power')
import numpy as np
from scipy.optimize import fsolve


def power_check(min, max):
    """
    Function to check domain of power law, ensures it conforms to assumptions
    Returns False with an error message if conditions not met, otherwise True

    'min'   : minimum of power law domain
    'max'   : maximum of power law domain
    """
    if min <= 0:
        print('power_functions.power_check ERROR: '
                + 'distribution minimum must be greater than 0')
        return False
    if max <= min:
        print('power_functions.power_check ERROR: max must be greater than min')
        return False

    return True


def power_normalize(alpha, min = 1, max = 10):
    """
    Function to determine normalization constant to ensure power law is
    a probability distribution function (pdf, integral over domain is 1)

    'alpha' : exponent of power law
    'min'   : minimum of power law normalization range
    'max'   : maximum of power law normalization range
    """
     # Check power law domain
    if power_check(min, max) == False:
        return False

    if round(alpha, 7) == -1: # Case for alpha = -1, integral(1/x) = ln|x|+c
        norm = 1 / np.log(max/min)
    else: # For all cases when alpha != -1 this is a standard power law integral
        a1 = alpha + 1
        norm = a1 / (np.float_power(max, a1) - np.float_power(min, a1))
    return norm


def power_pdf(x = False, alpha = -1, min = False, max = False):
    """
    Generate a power law distribution, unstable for negative numbers

    'x'     : domain over which to evaluate power law
    'alpha' : exponent of power law
    'min'   : minimum of power law normalization range
    'max'   : maximum of power law normalization range
    """
    #Deal with inputs
    if np.size(x) == 1:
        if x == False:
            print('power_functions.power_pdf( x, alpha, min, max) ERROR: '
                    + '"x" must be an array')
            return False
    else:
        if x.any() == False:
            print('power_functions.power_pdf( x, alpha, min, max) ERROR: '
                    + '"x" is False')
            return False
    if min == False:
        min = np.amin(x)
    if max == False:
        max = np.amax(x)
    if power_check(min, max): # Check power law domain
        return power_normalize(alpha, min, max) * np.float_power(x, alpha)
    else:
        return False


def power_random(alpha = -1, min = 1, max = 10, size = 1, rng = None):
    """
    Generate random (strictly positive) samples from a power-law distribution
    Based on: http://mathworld.wolfram.com/RandomNumber.html

    'alpha' : exponent of power law
    'min'   : minimum of power law normalization range
    'max'   : maximum of power law normalization range
    'size'  : number of samples of power law to generate
    'rng'   : numpy random number generator

    """
    if power_check(min, max) == False: # Check power law domain
        return False

    # Generate uniformly distributed random variable
    try:
        y = rng.random(size = size)
    except:
        y = np.random.random(size = size)

    # Uniformly distributed y converted into power law sampling
    if round(alpha, 7) == -1: # Case for alpha = -1 to avoid division by 0
        x = np.exp(np.log(min) + np.log(max/min) * y)
    else: # General case for power law distributions with alpha != -1
        a1 = alpha + 1
        min1 = np.float_power(min, a1)
        max1 = np.float_power(max, a1)
        base  = min1 + (max1 - min1) * y
        x = np.float_power(base, 1/a1)

    return x


def power_mean(alpha = -1, min = 1, max = 10):
    """
    Determine mean of a power law distribution

    'alpha' : exponent of power law
    'min'   : minimum of power law normalization range
    'max'   : maximum of power law normalization range
    """
    if power_check(min, max) == False: # Check power law domain
        return False

    if round(alpha, 7) == -1: # Case for alpha = -1 to avoid division by 0
        mean = (max - min) / np.log(max / min)
    elif round(alpha, 7) == -2: # Case for alpha = -2 to avoid division by 0
        mean = -np.log(max / min) / ((1/max) - (1/min))
    else:# General case for power law distributions with alpha != -1, -2
        a1 = alpha + 1
        a2 = alpha + 2
        mean = ((a1 / a2) * (np.float_power(max, a2) - np.float_power(min, a2))
                          / (np.float_power(max, a1) - np.float_power(min, a1)))

    #Check to make sure only one value is returned
    if hasattr(mean, '__len__'): # Checks for the len attribute (list, ndarray)
        mean = mean[0]

    return mean


def power_median(alpha = -1, min = 1, max = 10):
    """
    Determine median of a power law distribution

    'alpha' : exponent of power law
    'min'   : minimum of power law normalization range
    'max'   : maximum of power law normalization range
    """
    if power_check(min, max) == False: # Check power law domain
        return False

    if round(alpha, 7) == -1: # Case for alpha = -1 to avoid division by 0
        med = np.exp(0.5 * np.log(max*min))
    else: # General case for power law distributions with alpha != -1
        a1 = alpha + 1
        min1 = np.float_power(min, a1)
        max1 = np.float_power(max, a1)
        med = np.float_power(0.5 * (max1 + min1), 1/a1)

    #Check to make sure only one value is returned
    if hasattr(med, '__len__'): # Checks for the len attribute (list, ndarray)
        med = med[0]

    return med


def power_max_mean(alpha, mean, min, max_init):
    """
    Function to compute the maximum of a power law distribution given a mean

    'alpha'     : exponent of power law
    'mean'      : mean value of power law
    'min'       : minimum of power law normalization range
    'max_init'  : initial guess of maximum of power law normalization range
    """
    # Define functions for computing the max with special alpha conditions
    def max_roots_2(max, *args): # *args = [min, mean]
        return (np.log(max/args[0]) / ((1/args[0]) - (1/max))) - args[1]
    def max_roots_1(max, *args): # *args = [min, mean]
        return ((max - args[0]) / np.log(max/args[0])) - args[1]
    def max_roots(max, *args): # *args = [min, mean, alpha]
        a1 = args[2] + 1
        a2 = args[2] + 2
        return ((args[1] * np.float_power(max, a1))
                    - ((a1/a2) * np.float_power(max, a2))
                    + ((a1/a2) * np.float_power(args[0], a2))
                    - (args[1] * np.float_power(args[0], a1)))

    # Special case that can cause imaginary maximum
    if (alpha < -2) and (mean >= min * ((alpha+1)/(alpha+2))):
        min = mean * (((alpha+2)/(alpha+1)))
        max = sys.float_info.max
        #Check to make sure only one value is returned
        if hasattr(min, '__len__'): # Checks for len attribute (list, ndarray)
            min = min[0]
        return max, min

    # Solve for roots
    info = {}
    ier = 0
    max_test = max_init
    while (ier != 1) and (max_test < float('inf')) and (max_test > min):
        if round(alpha, 7) == -2:
            max, info, ier, msg = fsolve(max_roots_2, max_test,
                                         (min, mean), full_output=True)
        elif round(alpha, 7) == -1:
            max, info, ier, msg = fsolve(max_roots_1, max_test,
                                         (min, mean), full_output=True)
        else:
            max, info, ier, msg = fsolve(max_roots, max_test,
                                         (min, mean, alpha),
                                         full_output=True)
        max_test /= 10
    if ier != 1:
        max = max_init
    if hasattr(max, '__len__'): # Checks for the len attribute (list, ndarray)
        max = max[0]
    if hasattr(min, '__len__'): # Checks for the len attribute (list, ndarray)
        min = min[0]
    return max, min


def power_min_mean(alpha, mean, max, min_init):
    """
    Function to compute the minimum of a power law distribution given a mean

    'alpha'     : exponent of power law
    'mean'      : mean value of power law
    'max'       : maximum of power law normalization range
    'min_init'  : initial guess of minimum of power law normalization range
    """
    # Define functions for computing the max with special alpha conditions
    def min_roots_2(min, *args): # *args = [max, mean]
        return (np.log(args[0]/min) / ((1/min) - (1/args[0]))) - args[1]
    def min_roots_1(min, *args): # *args = [max, mean]
        return ((args[0] - min) / np.log(args[0]/min)) - args[1]
    def min_roots(min, *args): # *args = [max, mean, alpha]
        a1 = args[2] + 1
        a2 = args[2] + 2
        return (((a1/a2) * np.float_power(min, a2))
                - (args[1] * np.float_power(min, a1))
                - ((a1/a2) * np.float_power(args[0], a2))
                + (args[1] * np.float_power(args[0], a1)))

    # Special case that can cause minimum less than 0
    if (alpha > -1) and (mean <= max * ((alpha+1)/(alpha+2))):
        max = mean * (((alpha+2)/(alpha+1)))
        min = sys.float_info.min
        #Check to make sure only one value is returned
        if hasattr(max, '__len__'): # Checks for len attribute (list, ndarray)
            max = max[0]
        return max, min

    # Solve for roots
    info = {}
    ier = 0
    min_test = min_init
    while (ier != 1) and (min_test > 0) and (min_test < max):
        if round(alpha, 7) == -2: # Case for alpha = -2 to avoid divide by 0
            min, info, ier, msg = fsolve(min_roots_2, min_test,
                                         (max, mean), full_output=True)
        elif round(alpha, 7) == -1: # Case for alpha = -1 to avoid divide by 0
            min, info, ier, msg = fsolve(min_roots_1, min_test,
                                         (max, mean), full_output=True)
        else: # General case for power law distributions with alpha != -1, -2
            min, info, ier, msg = fsolve(min_roots, min_test,
                                         (max, mean, alpha),
                                         full_output=True)
        if hasattr(max, '__len__'): # Checks for len attribute (list, ndarray)
            max = max[0]
        if hasattr(min, '__len__'): # Checks for len attribute (list, ndarray)
            min = min[0]
        if round(min, 7) == round(max, 7):
            ier = 0
        min_test /= 10
    if ier != 1:
        min = min_init
    if hasattr(min, '__len__'): # Checks for the len attribute (list, ndarray)
        min = min[0]
    return max, min


def power_max_median(alpha = -1, median = 10, min = False):
    """
    Function to compute the maximum of a power law distribution given a median
    'alpha'     : exponent of power law
    'mean'      : median value of power law
    'min'       : minimum of power law normalization range
    'max_init'  : initial guess of maximum of power law normalization range
    """
    # Ensure median is greater than 0
    if median <= 0:
        print('power_functions.power_max_median ERROR: '
                + 'median must be greater than 0.')
        return False, False

    # Ensure minimum value is valid
    if min == False:
        min = 0.1 * median
    if min <= 0:
        print('power_functions.power_max_median ERROR: '
                + 'minimum must be greater than 0.')
        return False, False
    if min >= median:
        print('power_functions.power_max_median ERROR: '
                + 'minimum must be less than median.')
        return False, False
    if alpha < -1: # Special case that can cause imaginary maximum
        a1 = alpha + 1
        threshold = median * np.float_power(2, 1/a1)
        if min <= threshold: # condition where max -> inf
            min = threshold + 9e-9*median

    # Solve for maximum value
    if round(alpha, 7) == -1: # Special case for alpha = -1 to avoid divide by 0
        max = (1/min) * np.exp(2 * np.log(median))
    else: # General case for power law distributions with alpha != -1
        a1 = alpha + 1
        med1 = np.float_power(median, a1)
        min1 = np.float_power(min, a1)
        max = np.float_power(2 * med1 - min1, 1/a1)

    return max, min


def power_min_median(alpha = -1, median = 10, max = False):
    """
    Function to compute the maximum of a power law distribution given a median

    'alpha'     : exponent of power law
    'mean'      : median value of power law
    'min'       : minimum of power law normalization range
    'max_init'  : initial guess of maximum of power law normalization range
    """
    # Ensure median is greater than 0
    if median <= 0:
        print('power_functions.power_min_median ERROR: '
                + 'median must be greater than 0.')
        return False, False

    # Ensure maximum value is valid
    if max == False:
        max = 10 * median
    if max > sys.float_info.max :
        print('power_functions.power_min_median ERROR: '
                + 'maximum must be less than float limit (~1.8e308)')
        return False, False
    if max <= median:
        print('power_functions.power_min_median ERROR: '
                + 'maximum must be greater than median.')
        return False, False
    if alpha > -1: # Special case that can cause imaginary minimum
        a1 = alpha + 1
        threshold = median * np.float_power(2, 1/a1)
        if max >= threshold: # condition where min <= 0
            max = threshold - 9e-9*median

    # Solve for minimum value
    if round(alpha, 7) == -1: # Special case for alpha = -1 to avoid divide by 0
        min = (1/max) * np.exp(2 * np.log(median))
    else: # General case for power law distributions with alpha != -1
        a1 = alpha + 1
        med1 = np.float_power(median, a1)
        max1 = np.float_power(max, a1)
        min = np.float_power(2 * med1 - max1, 1/a1)

    return max, min


def power_domain(alpha = -1, median = 1, min = False, max = False,
                 mean = False):
    """
    Determine the domain of a power law distribution with a given central value
    'alpha'     : exponent of power law
    'median'    : median (or mean) of the power law
    'min'       : minimum of power law normalization range. Defers to 'max'.
    'max'       : maximum of power law normalization range. Overrides 'min'.
        NOTE    : if both 'min' and 'max' are provided, they are used as limits
                : and the largest domain possible is returned
    'mean'      : keyword toggle to compute the domain for a given mean instead
    """
    if min == False and max == False: # Calculate default power law
        print('power_functions.power_domain WARNING: '
                + 'no min or max supplied, assuming default')
        omax, omin = power_min_median(alpha, median) # Default min
        omax, omin = power_max_median(alpha, median, omin) # Default max
        if mean == True: # Need omax as a guess to calculate 'min' for 'mean'
            omax, omin = power_min_mean(alpha, median, omax, median)
            omax, omin = power_max_mean(alpha, median, omin, omax)
    elif max == False:
        omax, omin = power_max_median(alpha, median, min) # Calculate max
        if mean == True: # Need omax as a guess to calculate 'max' for 'mean'
            omax, omin = power_max_mean(alpha, median, min, omax)
    elif min == False:
        if mean == True:
            omax, omin = power_min_mean(alpha, median, max, median)
        else:
            omax, omin = power_min_median(alpha, median, max)
    else: # Both min and max are supplied
        if mean == True:
            if np.isnan(min):
                print('OUTSIDE', min, max, median, alpha)
            omax, omin = power_min_mean(alpha, median, max, median) # Use 'max'
            if omin < min: # If 'max' would imply a lower limit below 'min'
                omax, omin = power_max_mean(alpha, median, min, omax)
        else:
            omax, omin = power_min_median(alpha, median, max) # Assume max
            if omin < min: # If 'max' would imply a lower limit below 'min'
                omax, omin = power_max_median(alpha, median, min)

    return omax, omin


# Behaviour if called as a script
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    alpha = -1.1705#-2.1
    min = 50
    max = 86400
    pwr = power_random(alpha = alpha, min = min, max = max, size = 1000000)
    med = power_median(alpha = alpha, min = min, max = max)
    med = 3000
    mean = True
    alpha_test = np.arange(-3.0, 0.01, 0.01)
    max1 = []
    min1 = []
    for alpha_t in alpha_test:
        maxt, mint = power_domain(alpha = alpha_t, median = med, mean=mean,
                                    min=min, max=max)
        if mean:
            test = power_mean(alpha_t, mint, maxt)
        else:
            test = power_median(alpha_t, mint, maxt)
        if round(test, 7) == med:
            pass
        else:
            print(f"FAILURE: Alpha = {alpha_t}, Target = {med}, "
                    + f"test = {test}, {mint}, {maxt}")
        max1.append(maxt)
        min1.append(mint)

        if round(alpha_t, 7) == -2.5:
            print('Saving powerlaws')
            max300, min300 = power_domain(alpha = alpha_t, median = 300,
                                            mean=mean, min=min, max=max)
            max3000, min3000 = power_domain(alpha = alpha_t, median = 3000,
                                            mean=mean, min=min, max=max)
            tt300 = np.arange(min300, max300)
            tt3000 = np.arange(min3000, max3000)
            pdf300 = power_pdf(tt300, alpha_t, min300, max300)
            pdf3000 = power_pdf(tt3000, alpha_t, min3000, max3000)

            fig = plt.figure(figsize=(3.5,2))
            plt.plot(tt300, pdf300, label = 'Mean = 300 [s]')
            plt.plot(tt3000, pdf3000, label = 'Mean = 3000 [s]')
            plt.xlabel('Event delay time [s]')
            plt.legend(loc = 'best')
            plt.xscale('log')
            plt.yscale('log')
            plt.subplots_adjust(left = 0.15, bottom = 0.25)
            plt.savefig('powerlaw_300_3000_mean_distributions.png', dpi = 600)
            plt.clf()

    max = 3 * 60 * 60 # Three hour maximum delay
    min1 = 100
    min2 = 1000
    alpha1 = -2.5
    alpha2 = -1
    xx1 = np.arange(min1, max)
    xx2 = np.arange(min2, max)

    pdf1_1 = power_pdf(xx1, alpha1, min1, max) # distribution
    mean1_1 = power_mean(alpha1, min1, max) # distribution mean
    median1_1 = power_median(alpha1, min1, max) # distribution median
    pdf1_2 = power_pdf(xx1, alpha2, min1, max) # distribution
    mean1_2 = power_mean(alpha2, min1, max) # distribution mean
    median1_2 = power_median(alpha2, min1, max) # distribution median
    pdf2_1 = power_pdf(xx2, alpha1, min2, max) # distribution
    mean2_1 = power_mean(alpha1, min2, max) # distribution mean
    median2_1 = power_median(alpha1, min2, max) # distribution median
    pdf2_2 = power_pdf(xx2, alpha2, min2, max) # distribution
    mean2_2 = power_mean(alpha2, min2, max) # distribution mean
    median2_2 = power_median(alpha2, min2, max) # distribution median

    fig = plt.figure(figsize=(3.5,2))
    ax = fig.add_subplot(111)
    # Plot distribution lines
    plt.plot(xx1, pdf1_1, color = 'C0',
                label = r'Min = {0} [s], $\alpha$ = {1}'.format(min1, alpha1))
    plt.plot(xx1, pdf1_2, color = 'C1',
                label = r'Min = {0} [s], $\alpha$ = {1}'.format(min1, alpha2))
    plt.plot(xx2, pdf2_1, '--', color = 'C0',
                label = r'Min = {0} [s], $\alpha$ = {1}'.format(min2, alpha1))
    plt.plot(xx2, pdf2_2, '--', color = 'C1',
                label = r'Min = {0} [s], $\alpha$ = {1}'.format(min2, alpha2))

    plt.xlabel('Event delay time [s]')
    plt.ylabel('Event probability [1/s]')
    plt.legend(loc = 'lower left', prop = {'size': 6})
    plt.xscale('log')
    plt.yscale('log')
    plt.subplots_adjust(left = 0.2, bottom = 0.25)
    plt.savefig('powerlaw_distributions.png', dpi = 600)
