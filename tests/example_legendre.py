import numpy as np
import numpy.polynomial as P
import matplotlib.pyplot as plt

if __name__ == "__main__":
    L = 2
    xi = np.linspace(0, L, 100)
    # Generate x values from -1 to 1

    BC1_tag = "F"
    BC2_tag = "F"

    BC1 = 1
    BC2 = 1
    if BC1_tag == "F":
        BC1 = 1
    elif BC1_tag == "S":
        BC1 = (xi)
    elif BC1_tag == "C":
        BC1 = (xi)**2

    if BC2_tag == "F":
        BC2 = 1
    elif BC2_tag == "S":
        BC2 = (xi-L)
    elif BC2_tag == "C":
        BC2 = (xi-L)**2

    leg = P.legendre.Legendre((0,0,0,1), domain=[0, L], window=[-1, 1])
    leg = leg.convert(kind=P.polynomial.Polynomial)
    print(leg)

    y = BC1*BC2*leg(xi)

    ax = plt.plot(xi, y)

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('P_n(x)')
    plt.title('Legendre Polynomials')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()