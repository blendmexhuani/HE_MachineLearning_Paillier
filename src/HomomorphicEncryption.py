from numbers import Number
import numpy as np
import seal
from seal import ChooserEvaluator, \
    Ciphertext, \
    Decryptor, \
    Encryptor, \
    EncryptionParameters, \
    Evaluator, \
    IntegerEncoder, \
    FractionalEncoder, \
    KeyGenerator, \
    MemoryPoolHandle, \
    Plaintext, \
    SEALContext, \
    EvaluationKeys, \
    GaloisKeys, \
    PolyCRTBuilder, \
    ChooserEncoder, \
    ChooserEvaluator, \
    ChooserPoly


def initialize_fractional(
        poly_modulus_degree=4096,
        security_level_bits=128,
        plain_modulus_power_of_two=8,
        plain_modulus=None,
        encoder_integral_coefficients=1024,
        encoder_fractional_coefficients=3072,
        encoder_base=2
):
    parameters = EncryptionParameters()

	# Larger polynomial modulus makes the scheme more secure. At the same time, it
	# makes ciphertext sizes larger, and consequently all operations slower.
	# Recommended degrees for poly_modulus are 1024, 2048, 4096, 8192, 16384, 32768,
	# but it is also possible to go beyond this. Since we perform only a very small
	# computation in this example, it suffices to use a small polynomial modulus.
    poly_modulus = "1x^" + str(poly_modulus_degree) + " + 1"
    parameters.set_poly_modulus(poly_modulus)

    # A larger coefficient modulus also lowers the security level of the scheme. 
    # Thus, if a large noise budget is required for complicated computations, a large
    # coefficient modulus needs to be used, and the reduction in the security level 
    # must be countered by simultaneously increasing the polynomial modulus.
    if security_level_bits == 128:
        parameters.set_coeff_modulus(seal.coeff_modulus_128(poly_modulus_degree))
    elif security_level_bits == 192:
        parameters.set_coeff_modulus(seal.coeff_modulus_192(poly_modulus_degree))
    else:
        parameters.set_coeff_modulus(seal.coeff_modulus_128(poly_modulus_degree))
        print("Info: security_level_bits unknown - using default security_level_bits = 128")

    # The plaintext modulus can be any positive integer, even though here we take
	# it to be a power of two. The plaintext modulus determines the size of the
    # plaintext data type, but it also affects the noise budget in a freshly encrypted
    # ciphertext, and the consumption of the noise budget in homomorphic multiplication.
    # Thus, it is essential to try to keep the plaintext data type as small as possible 
    # for good performance.
    if plain_modulus is None:
        plain_modulus = 1 << plain_modulus_power_of_two
    parameters.set_plain_modulus(plain_modulus)

    # Now that all parameters are set, we are ready to construct a SEALContext
	# object. This is a heavy class that checks the validity and properties of
	# the parameters we just set, and performs and stores several important
	# pre-computations.
    context = SEALContext(parameters)
	
    # Print the parameters that we have chosen
    print_parameters(context)

    # The FractionalEncoder is used to compute a weighted average of 10 encrypted
    # rational numbers. In this computation we perform homomorphic multiplications 
    # of ciphertexts by plaintexts, which is much faster than regular multiplications
    # of ciphertexts by ciphertexts. Moreover, such `plain multiplications' never 
    # increase the ciphertext size.
    global encoder
    encoder = FractionalEncoder(
        context.plain_modulus(),
        context.poly_modulus(),
        encoder_integral_coefficients,
        encoder_fractional_coefficients,
        encoder_base
    )

    # Setting up the keys (public and private)
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()

    # We also set up an Encryptor, Evaluator, and Decryptor here.
    global encryptor
    encryptor = Encryptor(context, public_key)
    global evaluator
    evaluator = Evaluator(context)
    global decryptor
    decryptor = Decryptor(context, secret_key)
    
    # Since we are going to do some multiplications we will also relinearize.
    global evaluation_keys
    evaluation_keys = EvaluationKeys()
    keygen.generate_evaluation_keys(16, evaluation_keys)


class EncNum(Number):

    def __init__(self, encrypted):
        super().__init__()
        self.encrypted = encrypted

    def __add__(self, other):
        result = EncNum(Ciphertext(self.encrypted))
        if isinstance(other, EncNum):
            evaluator.add(result.encrypted, other.encrypted)
        else:
            other = float(other)
            other_plain = encoder.encode(other)
            evaluator.add_plain(result.encrypted, other_plain)
        return result

    __radd__ = __add__

    def __mul__(self, other):
        result = EncNum(Ciphertext(self.encrypted))
        if isinstance(other, EncNum):
            evaluator.multiply(result.encrypted, other.encrypted)
        else:
            other = float(other)
            if other == 0.0:
                raise ValueError("multiply_plain: plain cannot be zero")
            other_plain = encoder.encode(other)
            evaluator.multiply_plain(result.encrypted, other_plain)
        return result

    __rmul__ = __mul__

    def __sub__(self, other):
        result = EncNum(Ciphertext(self.encrypted))
        if isinstance(other, EncNum):
            evaluator.sub(result.encrypted, other.encrypted)
        else:
            other = float(other)
            other_plain = encoder.encode(other)
            evaluator.sub_plain(result.encrypted, other_plain)
        return result

    def __rsub__(self, other):
        if isinstance(other, EncNum):
            result = EncNum(Ciphertext(other.encrypted))
            evaluator.sub(result.encrypted, self.encrypted)
        else:
            result = EncNum(Ciphertext(self.encrypted))
            other = float(other)
            other_plain = encoder.encode(other)
            evaluator.sub_plain(result.encrypted, other_plain)
            evaluator.negate(result.encrypted)
        return result

    def __neg__(self):
        result = EncNum(Ciphertext(self.encrypted))
        evaluator.negate(result.encrypted)
        return result


def encrypt(n):
    plain = encoder.encode(n)
    encrypted = Ciphertext()
    encryptor.encrypt(plain, encrypted)
    return EncNum(encrypted)


encrypt_ndarray = np.vectorize(encrypt)


def decrypt(n):
    plain_result = Plaintext()
    decryptor.decrypt(n.encrypted, plain_result)
    return encoder.decode(plain_result)


decrypt_ndarray = np.vectorize(decrypt)


def recrypt(n):
    i_dec = decrypt(n)
    return encrypt(i_dec)


recrypt_ndarray = np.vectorize(recrypt)


def get_noise_budget(n):
    return decryptor.invariant_noise_budget(n.encrypted)


def print_parameters(context):
    print("| Encryption parameters:")
    print("|-- poly_modulus: " + context.poly_modulus().to_string())
    print("|-- coeff_modulus_size: " + str(context.total_coeff_modulus().significant_bit_count()) + " bits")
    print("|-- plain_modulus: " + str(context.plain_modulus().value()))
    print("|-- noise_standard_deviation: " + str(context.noise_standard_deviation()))
