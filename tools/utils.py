"""
Tools for the E2E communication system.

Acknowledgement:
The claude of Dr. Rasmus T. Jones.
The OptiCommPy-main of Dr. Edson Porto da Silva, etc.

Express my gratitude and respect here.

"""

import numpy as np
from scipy.optimize import fminbound
import torch
from numba import njit, prange
import logging as logg

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ''
        for key in self.__dict__.keys():
            s = s + key + ':' + 1*'\t' + str( self.__dict__[key] ) + '\n'

        return s

def unitlessAxis(samplesPerSymbol,filterSpan):
    axis = np.linspace(-(filterSpan/2), (filterSpan/2), samplesPerSymbol*filterSpan+1)
    return axis[:-1]

def freqAxis(N,Fs):
    if N/2%1==0:
        f = np.concatenate( [np.arange(0,N/2), np.arange(-N/2,0)] )/(N/Fs)
    else:
        f = np.concatenate( [np.arange(0,N/2), np.arange(-N/2+.5,0)] )/(N/Fs)
    return f


def omegaAxis(N,Fs):
    return 2*np.pi*freqAxis(N,Fs)


def hotOnes(size,transpose,M,seed=None):
    if seed!=None:
        np.random.seed(seed)
    x_seed = np.eye(M, dtype=int)
    idx = np.random.randint(M,size=size)
    x = np.transpose(x_seed[:,idx], transpose)
    return x, idx, x_seed


def lin2dB(lin,dBtype='dBm'):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    return 10*np.log10(lin)-fact


def dB2lin(dB, dBtype='dB'):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0.0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30.0
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60.0
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    return 10.0**( (dB+fact)/10.0 )

def SNRtoMI(N,effSNR,constellation):
    N = int(N)

    SNRlin = 10**(effSNR/10)
    constellation = constellation/np.sqrt(np.mean(np.abs(constellation)**2))
    M = constellation.size

    ## Simulation
    x_id = np.random.randint(0,M,(N,))
    x = constellation[:,x_id]

    z = 1/np.sqrt(2)*( np.random.normal(size=x.shape) + 1j*np.random.normal(size=x.shape) );
    y = x + z*np.sqrt(1/SNRlin);

    return calcMI_MC(x,y,constellation)

def calcMI_MC(x,y,constellation):
    """
        Transcribed from Dr. Tobias Fehenberger MATLAB code.
        See: https://www.fehenberger.de/#sourcecode
    """
    x = x.numpy()
    y = y.numpy()
    constellation = constellation.numpy()

    if y.shape[0] != 1:
        y = y.T
    if x.shape[0] != 1:
        x = x.T
    if constellation.shape[0] == 1:
        constellation = constellation.T

    M = constellation.size
    N = x.size
    P_X = np.zeros( (M,1) )

    x = x / np.sqrt( np.mean( np.abs( x )**2 ) ) # normalize such that var(X)=1
    y = y / np.sqrt( np.mean( np.abs( y )**2 ) ) # normalize such that var(Y)=1

    ## Get X in Integer Representation
    xint = np.argmin( np.abs( x - constellation )**2, axis=0)

    fun = lambda h: np.dot( h*x-y, np.conj( h*x-y ).T )
    h = fminbound( fun, 0,2)
    N0 = np.real( (1-h**2)/h**2 )
    y = y / h

    ## Find constellation and empirical input distribution
    for s in np.arange(M):
        P_X[s] = np.sum( xint==s ) / N
        
    ## Monte Carlo estimation of (a lower bound to) the mutual information I(XY)
    qYonX = 1 / ( np.pi*N0 ) * np.exp( ( -(np.real(y)-np.real(x))**2 -(np.imag(y)-np.imag(x))**2 ) / N0 )
    
    qY = 0
    for ii in np.arange(M):
        qY = qY + P_X[ii] * (1/(np.pi*N0)*np.exp((-(np.real(y)-np.real(constellation[ii,0]))**2-(np.imag(y)-np.imag(constellation[ii,0]))**2)/N0))
    
    realmin = np.finfo(float).tiny
    MI=1/N*np.sum(np.log2(np.maximum(qYonX,realmin)/np.maximum(qY,realmin)))

    return MI

def gaussianMI(x, y, constellation, M, dtype=torch.float64):
    """
    Computes mutual information with Gaussian auxiliary channel assumption and constellation with uniform probability distribution

    x: (1, N), N normalized complex samples at the transmitter, where N is the batchSize/sampleSize
    y: (1, N), N normalized complex observations at the receiver, where N is the batchSize/sampleSize
    constellation: (1, M), normalized complex constellation of order M

    Transcribed from Dr. Tobias Fehenberger MATLAB code.
    """
    if len(constellation.shape) == 1:
        constellation = constellation.unsqueeze(0)
    if len(y.shape) == 1:
        y = y.unsqueeze(0)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    if y.shape[0] != 1:
        y = y.T
    if x.shape[0] != 1:
        x = x.T
    if constellation.shape[0] == 1:
        constellation = constellation.T

    N = x.shape[1]
    PI = torch.tensor(np.pi, dtype=dtype)
    REALMIN = torch.tensor(np.finfo(float).tiny, dtype=dtype)

    # Find the closest constellation points
    xint = torch.argmin(torch.abs(x - constellation)**2, dim=0)
    x_count = torch.bincount(xint, minlength=M)
    P_X = x_count.float() / N

    # Compute noise power
    N0 = torch.mean(torch.abs(x - y)**2)

    # Calculate q(y|x)
    real_diff = (torch.real(y) - torch.real(x))**2
    imag_diff = (torch.imag(y) - torch.imag(x))**2
    qYonX = 1 / (PI * N0) * torch.exp(-(real_diff + imag_diff) / N0)

    # Compute q(y)
    qY = torch.zeros_like(qYonX)
    for ii in range(M):
        real_diff = (torch.real(y) - torch.real(constellation[ii, 0]))**2
        imag_diff = (torch.imag(y) - torch.imag(constellation[ii, 0]))**2
        temp = P_X[ii] * (1 / (PI * N0) * torch.exp(-(real_diff + imag_diff) / N0))
        qY += temp

    # Calculate mutual information (MI)
    MI = (1 / N) * torch.sum(torch.log2(torch.max(qYonX, REALMIN) / torch.max(qY, REALMIN)))

    return MI


def generateBitVectors(N, M):
    # Generates N bit vectors with log2(M) bits
    w = int(np.log2(M))
    d = np.zeros((N, w))
    r = np.random.randint(low=0, high=M, size=(N,))
    for ii in range(N):
        d[ii,:] = np.array( [ float(x) for x in np.binary_repr(r[ii], width=w) ] )
    return d


def generateUniqueBitVectors(M):
    # Generates log2(M) unique bit vectors with M bits
    w = int(np.log2(M))
    d = np.zeros((M,w))
    for ii in range(M):
        d[ii,:] = np.array( [ float(x) for x in np.binary_repr(ii,width=w) ] )
    return d



def complex2real(x, axis=-1, dtype=torch.float32):
    real_part = torch.real(x)
    imag_part = torch.imag(x)
    ret = torch.stack((real_part, imag_part), dim=axis)
    ret = ret.squeeze(1).to(dtype)
    return ret


def norm_factor(constellation, epsilon=torch.tensor(1e-12)):
    if any([constellation.dtype == x for x in [torch.complex64, torch.complex128]]):
        castTo = constellation.dtype
        constellation = complex2real(constellation)
    else:
        castTo = False

    rmean = torch.mean(torch.square(torch.norm(constellation, dim=-1)))
    normFactor = torch.rsqrt(torch.maximum(rmean, epsilon))

    if castTo:
        return normFactor.type_as(constellation)
    else:
        return normFactor


# MI and GMI calculation
def calcLLR(rxSymb, σ2, constSymb, bitMap, px):
    """
    LLR calculation (circular AGWN channel).

    Parameters
    ----------
    rxSymb : np.array
        Received symbol sequence.
    σ2 : scalar
        Noise variance.
    constSymb : (M, 1) np.array
        Constellation symbols.
    px : (M, 1) np.array
        Prior symbol probabilities.
    bitMap : (M, log2(M)) np.array
        Bit-to-symbol mapping.

    Returns
    -------
    LLRs : np.array
        sequence of calculated LLRs.

    References
    ----------
    [1] A. Alvarado, T. Fehenberger, B. Chen, e F. M. J. Willems, “Achievable Information Rates for Fiber Optics: Applications and Computations”, Journal of Lightwave Technology, vol. 36, nº 2, p. 424–439, jan. 2018, doi: 10.1109/JLT.2017.2786351.

    """
    M = len(constSymb)
    b = int(np.log2(M))

    LLRs = np.zeros(len(rxSymb) * b)

    for i in prange(len(rxSymb)):
        prob = np.exp((-np.abs(rxSymb[i] - constSymb) ** 2) / σ2) * px

        for indBit in range(b):
            p0 = np.sum(np.array(prob[bitMap[:, indBit] == 0]))
            p1 = np.sum(np.array(prob[bitMap[:, indBit] == 1]))

            LLRs[i * b + indBit] = np.log(p0) - np.log(p1)
    return LLRs


def monteCarloGMI(rx, tx, M, constType='qam', px=None, constellation=None, constellation_bits=None, txBits=None):
    """
    Monte Carlo based generalized mutual information (GMI) estimation.

    Parameters
    ----------
    rx : np.array
        Received symbol sequence.
    tx : np.array
        Transmitted symbol sequence.
    M : int
        Modulation order.
    constType : string
        Modulation type: 'qam' or 'psk'    (matrix)
    px : (M, 1) np.array
        Prior symbol probabilities. The default is [].
    constellation : (M, ) np.array
        complex constellation points.
    constellation_bits : (M, m) np.array
        bits mapping for constellation points.

    Returns
    -------
    GMI : np.array
        Generalized mutual information values.
    NGMI : np.array
        Normalized mutual information.

    References
    ----------
    [1] A. Alvarado, T. Fehenberger, B. Chen, e F. M. J. Willems, “Achievable Information Rates for Fiber Optics: Applications and Computations”, Journal of Lightwave Technology, vol. 36, nº 2, p. 424–439, jan. 2018, doi: 10.1109/JLT.2017.2786351.

    """
    if px is None:
        px = []
    # constellation parameters
    constSymb = grayMapping(M, constType)

    # get bit mapping
    b = int(np.log2(M))
    bitMap = demodulateGray(constSymb, M, constType)
    bitMap = bitMap.reshape(-1, b)

    # mapping for Encoder
    if constellation_bits != None:
        bitMap = np.array(constellation_bits)

    # We want all the signal sequences to be disposed in columns:
    try:
        if rx.shape[1] > rx.shape[0]:
            rx = rx.T
    except IndexError:
        rx = rx.reshape(len(rx), 1)
    try:
        if tx.shape[1] > tx.shape[0]:
            tx = tx.T
    except IndexError:
        tx = tx.reshape(len(tx), 1)
    nModes = int(tx.shape[1])  # number of sinal modes
    GMI = np.zeros(nModes)
    NGMI = np.zeros(nModes)

    if len(px) == 0:  # if px is not defined, assume uniform distribution
        px = 1 / M * np.ones(constSymb.shape)
    # Normalize constellation
    Es = np.sum(np.abs(constSymb) ** 2 * px)
    constSymb = constSymb / np.sqrt(Es)
    # data from constellation points
    if constellation != None:
        constSymb = np.array(constellation)

    # Calculate source entropy
    H = np.sum(-px * np.log2(px))

    # symbol normalization
    for k in range(nModes):
        if constType in ["qam", "psk"]:
            # correct (possible) phase ambiguity
            # rot = np.mean(tx[:, k] / rx[:, k])
            rot = np.mean(np.array(tx[:, k] / rx[:, k]))
            rot = torch.tensor(rot, dtype=torch.complex64)  # must be torch.tensor to avoid lost imaginary part
            rx[:, k] = rot * rx[:, k]
        # symbol normalization
        rx[:, k] = pnorm(rx[:, k])
        tx[:, k] = pnorm(tx[:, k])
    for k in range(nModes):
        # set the noise variance
        # σ2 = np.var(rx[:, k] - tx[:, k], axis=0)
        σ2 = np.var(np.array(rx[:, k] - tx[:, k]), axis=0)

        # demodulate transmitted symbol sequence
        btx = demodulateGray(np.sqrt(Es) * tx[:, k], M, constType)
        if txBits != None:
            btx = np.array(txBits)

        # soft demodulation of the received symbols
        LLRs = calcLLR(rx[:, k], σ2, constSymb, bitMap, px)

        # LLR clipping
        LLRs[LLRs == np.inf] = 500
        LLRs[LLRs == -np.inf] = -500

        # Compute bitwise MIs and their sum
        b = int(np.log2(M))

        MIperBitPosition = np.zeros(b)

        for n in range(b):
            MIperBitPosition[n] = H / b - np.mean(
                np.log2(1 + np.exp((2 * btx[n::b] - 1) * LLRs[n::b]))
            )
        GMI[k] = np.sum(MIperBitPosition)
        NGMI[k] = GMI[k] / H
    return GMI, NGMI


def demodulateGray(symb, M, constType):
    """
    Demodulate symbol sequences to bit sequences (w/ Gray mapping).

    Hard demodulation is based on minimum Euclidean distance.

    Parameters
    ----------
    symb : array of complex constellation symbols
        sequence of constellation symbols to be demodulated.
    M : int
        order of the modulation format.
    constType : string
        'qam', 'psk', 'apsk', 'pam' or 'ook'.

    Returns
    -------
    array of ints
        sequence of demodulated bits.     1-D array

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

    """
    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    const = grayMapping(M, constType)

    # get bit to symbol mapping
    indMap = minEuclid(const, const)
    bitMap = dec2bitarray(indMap, int(np.log2(M)))
    b = int(np.log2(M))
    bitMap = bitMap.reshape(-1, b)

    # demodulate received symbol sequence
    indrx = minEuclid(symb, const)

    return demap(indrx, bitMap)


def pnorm(x):
    """
    Normalize the average power of each componennt of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    np.array
        Signal x with each component normalized in power.

    """
    # return x / np.sqrt(np.mean(x * np.conj(x)).real)
    return x / np.sqrt(np.mean(np.array(x * np.conj(x))).real)


def grayMapping(M, constType):
    """
    Gray Mapping for digital modulations.

    Parameters
    ----------
    M : int
        modulation order
    constType : 'qam', 'psk', 'pam' or 'ook'.
        type of constellation.

    Returns
    -------
    const : np.array
        constellation symbols (sorted according their corresponding
        Gray bit sequence as integer decimal).

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2

    bitsSymb = int(np.log2(M))

    code = grayCode(bitsSymb)
    if constType == "ook":
        const = np.arange(0, 2)
    elif constType == "pam":
        const = pamConst(M)
    elif constType == "qam":
        const = qamConst(M)
    elif constType == "psk":
        const = pskConst(M)
    elif constType == "apsk":
        const = apskConst(M)

    const = const.reshape(M, 1)
    const_ = np.zeros((M, 2), dtype=complex)

    for ind in range(M):
        const_[ind, 0] = const[ind, 0]  # complex constellation symbol
        const_[ind, 1] = int(code[ind], 2)  # mapped bit sequence (as integer decimal)

    # sort complex symbols column according to their mapped bit sequence (as integer decimal)
    const = const_[const_[:, 1].real.argsort()]
    const = const[:, 0]

    if constType in ["pam", "ook"]:
        const = const.real
    return const


def grayCode(n):
    """
    Gray code generator.

    Parameters
    ----------
    n : int
        length of the codeword in bits.

    Returns
    -------
    code : list
           list of binary strings of the gray code.

    """
    code = []

    for i in range(1 << n):
        # Generating the decimal
        # values of gray code then using
        # bitset to convert them to binary form
        val = i ^ (i >> 1)

        # Converting to binary string
        s = bin(val)[2::]
        code.append(s.zfill(n))
    return code


def pamConst(M):
    """
    Generate a Pulse Amplitude Modulation (PAM) constellation.

    Parameters
    ----------
    M : int
        Number of symbols in the constellation. It must be an integer.

    Returns
    -------
    np.array
        1D PAM constellation.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    L = int(M - 1)
    return np.arange(-L, L + 1, 2)


def qamConst(M):
    """
    Generate a Quadrature Amplitude Modulation (QAM) constellation.

    Parameters
    ----------
    M : int
        Number of symbols in the constellation. It must be a perfect square.

    Returns
    -------
    const : np.array
        Complex square M-QAM constellation.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    L = int(np.sqrt(M) - 1)

    # generate 1D PAM constellation
    PAM = np.arange(-L, L + 1, 2)
    PAM = np.array([PAM])

    # generate complex square M-QAM constellation
    const = np.tile(PAM, (L + 1, 1))
    const = const + 1j * np.flipud(const.T)

    for ind in np.arange(1, L + 1, 2):
        const[ind] = np.flip(const[ind], 0)

    return const


def pskConst(M):
    """
    Generate a Phase Shift Keying (PSK) constellation.

    Parameters
    ----------
    M : int
        Number of symbols in the constellation. It must be a power of 2 positive integer.

    Returns
    -------
    np.array
        Complex M-PSK constellation.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    # generate complex M-PSK constellation
    pskPhases = np.arange(0, 2 * np.pi, 2 * np.pi / M)
    return np.exp(1j * pskPhases)


def apskConst(M, m1=None, phaseOffset=None):
    """
    Generate an Amplitude-Phase Shift Keying (APSK) constellation.

    Parameters
    ----------
    M : int
        Constellation order.
    m1 : int
        Number of bits used to index the radii of the constellation.

    Returns
    -------
    const : np.array
        APSK constellation

    References
    ----------
    [1] Z. Liu, et al "APSK Constellation with Gray Mapping," IEEE Communications Letters, vol. 15, no. 12, pp. 1271-1273, 2011.

    [2] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    if m1 is None:
        if M == 16:
            m1 = 1
        elif M == 32:
            m1 = 2
        elif M == 64:
            m1 = 2
        elif M == 128:
            m1 = 3
        elif M == 256:
            m1 = 3
        elif M == 512:
            m1 = 4
        elif M == 1024:
            m1 = 4

    nRings = int(2 ** m1)  # bits that index the rings
    m2 = int(np.log2(M) - m1)  # bits that index the symbols per ring

    symbolsPerRing = int(2 ** m2)

    const = np.zeros((M,), dtype=np.complex64)

    if phaseOffset is None:
        phaseOffset = np.pi / symbolsPerRing

    for idx in range(nRings):
        radius = np.sqrt(-np.log(1 - ((idx + 1) - 0.5) * symbolsPerRing / M))

        if (idx + 1) % 2 == 1:
            const[idx * symbolsPerRing: (idx + 1) * symbolsPerRing] = radius * np.flip(
                pskConst(symbolsPerRing)
            )
        else:
            const[
            idx * symbolsPerRing: (idx + 1) * symbolsPerRing
            ] = radius * pskConst(symbolsPerRing)

    return const * np.exp(1j * phaseOffset)


def minEuclid(symb, const):
    """
    Find minimum Euclidean distance.

    Find closest constellation symbol w.r.t the Euclidean distance in the
    complex plane.

    Parameters
    ----------
    symb : np.array
        Received constellation symbols.
    const : np.array
        Reference constellation.

    Returns
    -------
    np.array of int
        indexes of the closest constellation symbols.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

    """
    ind = np.zeros(symb.shape, dtype=np.int64)
    for ii in prange(len(symb)):
        ind[ii] = np.abs(symb[ii] - const).argmin()
    return ind


def dec2bitarray(x, bit_width):
    """
    Converts a positive integer or an array-like of positive integers to a NumPy array of the specified size containing
    bits (0 and 1).

    Parameters
    ----------
    x : int or array-like of int
        Positive integer(s) to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 2D NumPy array of int
        Array containing the binary representation of all the input decimal(s).

    """

    if isinstance(x, int):
        return decimal2bitarray(x, bit_width)
    result = np.zeros((len(x), bit_width), dtype=np.int64)
    for pox, number in enumerate(x):
        result[pox] = decimal2bitarray(number, bit_width)
    return result


def decimal2bitarray(x, bit_width):
    """
    Converts a positive integer to a NumPy array of the specified size containing bits (0 and 1). This version is slightly
    quicker but only works for one integer.

    Parameters
    ----------
    x : int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D NumPy array of int
        Array containing the binary representation of the input decimal.

    """
    result = np.zeros(bit_width, dtype=np.int64)
    i = 1
    pox = 0
    while i <= x:
        if i & x:
            result[bit_width - pox - 1] = 1
        i <<= 1
        pox += 1
    return result


def demap(indSymb, bitMap):
    """
    Contellation symbol index to bit sequence demapping.

    Parameters
    ----------
    indSymb : np.array of ints
        Indexes of received symbol sequence.
    bitMap : (M, log2(M)) np.array
        bit-to-symbol mapping.

    Returns
    -------
    decBits : np.array
        Sequence of demapped bits.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

    """
    M = bitMap.shape[0]
    b = int(np.log2(M))

    decBits = np.zeros(len(indSymb) * b, dtype="int")

    for i in prange(len(indSymb)):
        decBits[i * b: i * b + b] = bitMap[indSymb[i], :]
    return decBits
