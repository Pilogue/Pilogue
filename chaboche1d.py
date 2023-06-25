# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import xlwt


class Chaboche1D:
    """


    Args:
        param1 (array): Array containing the material parameters.

    """

    def __init__(self, E, K, n, H, D, h, d):
        self.E = E
        self.K = K
        self.n = n
        self.H = H
        self.D = D
        self.h = h
        self.d = d
        self.solutions = []

    def total_strain (self, t):
        """
        
        Args:
            param1 (int): Current moment of time(t).

         Returns:
            The mechanical displacement for the current moment
            of time(t).
        """
        tc = 20.0         # Cyclic time for one cyclic loading
        Emax = 0.036      # Maximum mechanical displacement
        Emin = -Emax
        tcicle = t-tc*math.floor(t/tc)
        # Calculate total strain
        if tcicle <= tc/4.0:
            return 4.0*(Emax/tc)*tcicle
        if tc/4.0 < tcicle <= (3.0/4.0)*tc:
            return (-(4.0 * Emax) / tc) * tcicle + 2.0 * Emax
        if (3.0/4.0)*tc < tcicle <= tc:
            return ((-4.0*Emin)/tc)*tcicle+4.0*Emin

    def deriv(self, z, t, ET):
        """
    
        Args:
            param1 (np.array): Array containing the values of viscoplastic
                            strain,back stress and drag stress.
            param2 (np.array): A sequence of time points for which 
                               to solve for z.
            param3 (int): The mechanical displacement for this step
        Returns:
            Array containing the derivatives of viscoplastic strain,
            back stress and drag stress in t, with the initial
            value z0 in the first row.
    
        """
        Evp = z[0]            # Viscoplastic strain
        X = z[1]              # Back stress
        R = z[2]              # Drag stress
        S = self.E*(ET-Evp)   # Calculate Total Stress
        if abs(S-X)-R < 0:    # Elastic state Von Mises 屈服判断
            dEvpdt = 0.
            dXdt = 0.
            dRdt = 0.
        else:              # Plastic state
            dEvpdt = (((abs(S-X)-R)/self.K)**self.n)*np.sign(S-X)
            dXdt = self.H*dEvpdt-self.D*X*abs(dEvpdt)
            dRdt = self.h*abs(dEvpdt)-self.d*R*abs(dEvpdt)
        return [dEvpdt, dXdt, dRdt]

    def solve(self, z0, t):
        """
        
        Args:
            paraml (np.array):Array containing the initial conditions.
            param2 (int):Total sequence of time points for which
                        to solve for z.
                        
        """
        # Iterate through the sequence of time points
        for i in range(1, len(t)):
            # Mechanical displacement for next step
            ET = self.total_strain(t[i])
            # Time span for the next time step
            tspan = [t[i-1], t[i]]
            # Solve for next step
            z = odeint(self.deriv, z0, tspan, args=(ET, ))
            rate.append(self.deriv(z0, tspan, ET))
            # Store solution
            self.solutions.append(z)
            # Next initial condition
            z0 = z[1]


# Main program


if __name__ == "__main__":
    # initial conditions−Evp/X/R
    z0 = [0, 0, 50.0]
    rate = [[0, 0, 0]]
    # number of data points
    n = 3000
    # Define material parameters
    # E,K,n,H,D,h,d
    model_1D = Chaboche1D(5000.0, 50.0, 3.0, 5000.0, 100.0, 300.0, 0.6)
    # Time points
    t = np.linspace(0, 80, n)
    # Solve Chaboche’s 1D model with given material parameters
    model_1D.solve(z0, t)
    ################# before standardization ###################
    data1 = np.array(model_1D.solutions)
    data = np.zeros((3000, 3))
    for i in range(0, len(t)):
        if i < n-1:
            data[i] = data1[i, 0]
        else:
            data[i] = data1[i-1, 1]
    sigma = [0]
    for i in range(1, len(t)):
        sigma.append(model_1D.E * (model_1D.total_strain(t[i]) - data[i, 0]))
    sigma1 = np.array(sigma)
    data = np.insert(data, 3, sigma1, axis=1)
    plt.plot(range(n), data[:, 0], label='Evp')
    plt.plot(range(n), data[:, 1], label='X')
    plt.plot(range(n), data[:, 2], label='R')
    plt.plot(range(n), data[:, 3], label='σ')
    plt.title("1D_raw")
    plt.xlabel("Training sample")
    plt.ylabel("stress/Mpa")
    plt.grid()
    plt.legend()
    plt.show()
    ################# standardization ###################
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data)
    plt.plot(range(n), data_train[:, 0], label='Evp')
    plt.plot(range(n), data_train[:, 1], label='X')
    plt.plot(range(n), data_train[:, 2], label='R')
    plt.plot(range(n), data_train[:, 3], label='σ')
    plt.title("1D_Standardization")
    plt.xlabel("Training sample")
    plt.ylabel("stress/Mpa")
    plt.grid()
    plt.legend()
    plt.show()

    # rate
    rate_data1 = np.array(rate)
    rate_data = np.zeros((3000, 3))
    for i in range(len(t)):
        rate_data[i] = rate_data1[i]
    plt.plot(range(n), rate_data[:, 0], label='rEvp')
    plt.plot(range(n), rate_data[:, 1], label='rX')
    plt.plot(range(n), rate_data[:, 2], label='rR')
    plt.title("1D_rate")
    plt.xlabel("Training sample")
    plt.ylabel("stress_rate/Mpa")
    plt.grid()
    plt.legend()
    plt.show()
    rate_train = scaler.fit_transform(rate_data)
    plt.plot(range(n), rate_train[:, 0], label='rEvp')
    plt.plot(range(n), rate_train[:, 1], label='rX')
    plt.plot(range(n), rate_train[:, 2], label='rR')
    plt.title("1D_Standardization_rate")
    plt.xlabel("Training sample")
    plt.ylabel("stress_rate/Mpa")
    plt.grid()
    plt.legend()
    plt.show()

    # Extract data before Standardization
    work_book = xlwt.Workbook(encoding="UTF-8")
    worksheet = work_book.add_sheet(sheetname='1d')
    for i in range(len(data)):
        for j in range(len(data[i]) + len(rate_data[i])):
            if j < len(data[i]):
                worksheet.write(i, j, data[i][j])
            else:
                worksheet.write(i, j, rate_data[i][j - len(data[i])])
    savePath = 'F:\\Coderlife\\Pilogue\\1draw.csv'
    work_book.save(savePath)

    # Extract data after Standardization
    work_book = xlwt.Workbook(encoding="UTF-8")
    worksheet = work_book.add_sheet(sheetname='1d')
    for i in range(len(data_train)):
        for j in range(len(data_train[i]) + len(rate_train[i])):
            if j < len(data_train[i]):
                worksheet.write(i, j, data_train[i][j])
            else:
                worksheet.write(i, j, rate_train[i][j - len(data_train[i])])
    savePath = 'F:\\Coderlife\\Pilogue\\1d.csv'
    work_book.save(savePath)


