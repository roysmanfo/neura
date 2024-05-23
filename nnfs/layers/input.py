



def Input(shape: tuple[int, ...]) -> int:
    """
    Define the input shape of the neural network
    """
    x = 1
    for i in shape:
        x *= i
    return x
