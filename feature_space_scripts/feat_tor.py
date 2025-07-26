#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:27:33 2023

@author: yanbin
"""
import numpy as np
#import matplotlib.pyplot as plt

def file_reader(filename):
    # Initialize an empty list to store the data
    data_list = []
    # Open the file and read the data
    with open(filename, 'r') as file:
        #next(file)  # Skip the header line
        for i, line in enumerate(file):
            #if i >= 0:  # Skip the first 5 rows after the header
                # Convert each line to a list of floats and append to the list
                data_list.append([float(x) for x in line.split()])
    # Convert the list to a NumPy array
    data = np.array(data_list)
    return data

N_sub = 4056                                                                                                                                                                                   
phipsi = file_reader('COLVAR')
phipsi_data = phipsi[:,1:3]
points = np.array(phipsi_data[:N_sub, :])  # Convert to numpy array
Mii = file_reader('Mii.dat')
Mij = file_reader('Mij.dat')
Mjj = file_reader('Mjj.dat')
Mii = np.array(Mii[:N_sub, 1])  # Convert to numpy array
Mij = np.array(Mij[:N_sub, 1])  # Convert to numpy array
Mjj = np.array(Mjj[:N_sub, 1])  # Convert to numpy array
print(points[0,:])
print(Mii[0])
batch = 100

try:
    weights = file_reader('weight-out-only')
    weights = np.array(weights[:N_sub, :])
    weights = weights.squeeze()
    weights = weights*(N_sub/np.sum(weights))
    print(np.sum(weights))
except FileNotFoundError:
    weights = np.ones_like(Mii)
    
def pbc_feature(diff):
    return (diff + np.pi) % (2 * np.pi) - np.pi

def distance(x1,x2):
    ndata_batch = x1.shape[0]
    ndim = x1.shape[1]
    diff = x1.reshape(ndata_batch,1,ndim)-x2
    diff_pbc = pbc_feature(diff)
    return diff_pbc
    
def gaussianKer(x1,x2,s):
    pre = -0.25/s
    diff_pbc = distance(x1,x2)
    dist = np.sum(diff_pbc**2,axis=2)
    #gau_ker = np.exp(pre*dist)/np.sqrt(2*np.pi*s)
    gau_ker = np.exp(pre*dist)/(2*np.pi*s)
    return gau_ker

def tKer(x1,x2,s):
    ndata_batch = x1.shape[0]
    ndim = x1.shape[1]
    pre = 1/s
    diff = x1.reshape(ndata_batch,1,ndim)-x2
    diff_pbc = pbc_feature(diff)
    dist = np.sum(diff_pbc**2,axis=2)
    t_ker = 1.0/(1+pre*dist)
    return t_ker

def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def calcProb(x, s, scan_size, w):
    xb_list = batch_data(x, scan_size)
    
    gau_ker_tot = np.zeros_like(w)
    for j, xb in enumerate(xb_list):
        # Calculate the Gaussian kernel (requires implementation)
        gau_ker = gaussianKer(xb, x, s)
        prob_batch = np.sum(gau_ker, axis=1) # weights\
        gau_ker_tot[j * scan_size:j * scan_size + prob_batch.shape[0]] = prob_batch
        
    prob_unbias = (gau_ker_tot*w)/np.sum(w)
    prob_unbias = prob_unbias/np.sum(prob_unbias)
    print(np.sum(prob_unbias))
    return prob_unbias
        
def mahalanobisKer(points, s, sum_m,det_m,weights): # it is the gaussion kernel with mahalanobis distance
    pre = -0.25 / s
    #prob = calcProb(points,0.1,1,weights)
    prob = calcProb(points,s,batch,weights) # here the prob is the true prob on biased data, not the biased prob on biased data
    prefactors = (weights*det_m**-0.25)/np.sqrt(prob)
    pre_matrix = np.tile(prefactors[np.newaxis, :], (prefactors.shape[0], 1)) # extend the prefactors of the kernel to square matrix
    
    N = points.shape[0]
    data_list = batch_data(points,batch)
    diff_pbc = np.zeros((N,N,2))
    for j, xb in enumerate(data_list):
        # Calculate the Gaussian kernel (requires implementation)
        diff_pbc_batch= distance(xb, points)
        diff_pbc[j * batch:j * batch + diff_pbc_batch.shape[0]] = diff_pbc_batch
    
    results = np.zeros((N, N))
    for index, V in enumerate(diff_pbc):
        M = sum_m[index, :]
        for i in range(N):
            V_i = V[i, :]
            #print(V)
            M_i = M[i, :, :]
            result = V_i.T @ M_i @ V_i
            #results[index, i] = np.exp(result * pre)/(2*np.pi)# normlization here
            results[index, i] = np.exp(result * pre)
    results = results*pre_matrix # denominator of expression
    norm_f = np.sum(results, axis=1, keepdims=True)
    Ker = results / norm_f  # Normalize the results, it is the approx Mij 
    D = norm_f.squeeze()
    D2 = np.sqrt(D*prefactors) # here the *prefactors is to cancel out the non-kernel terms in denominator to keep the form same as original DM
    return Ker,D2

def M_construction(Mii, Mij, Mjj):
    N = len(Mii)
    M_3D = np.zeros((N, 2, 2))
    det_m = np.zeros(N)
    for index in range(N):
        M = np.array([[Mii[index], Mij[index]], [Mij[index], Mjj[index]]])
        det_m[index] = np.linalg.det(M)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:  # Correct exception type for NumPy
            print(f"Matrix at index {index} is singular and cannot be inverted.")
            M_inv = np.zeros((2, 2))
        M_3D[index, :, :] = M_inv

    # Adjusting for broadcasting instead of unsqueeze
    M1 = M_3D[:, np.newaxis, :, :]
    M2 = M_3D[np.newaxis, :, :, :]
    sum_m = M1 + M2
    return sum_m, det_m

def symEig(M, D, ndiml):
    Ms = D.reshape(-1, 1) * M / D.reshape(1, -1)
    v, w = np.linalg.eigh(Ms)
    v_out = v[-ndiml:]
    w_out = w[:, -ndiml:]
    w_r = w_out / D.reshape(-1, 1)
    w_r_normalized = w_r / np.linalg.norm(w_r, axis=0)
    w_l = w_out * D.reshape(-1, 1)
    # Save directly as NumPy arrays
    np.savetxt('eig-vec-r', w_r_normalized)
    np.savetxt('eig-vec-l', w_l)
    return v_out, w_r, w_l
        
s = 0.1 # sigma
M_c = M_construction(Mii, Mij, Mjj)
maha_results = mahalanobisKer(points,s,M_c[0],M_c[1],weights)
maha_Ker = maha_results[0]
D2 =  maha_results[1]

row_m = np.sum(maha_Ker,axis=0)
col_m = np.sum(maha_Ker,axis=1)
v,w_r,w_l= symEig(maha_Ker,D2,5)
check = np.sqrt(np.sum(w_r[:,1]**2))
print(check)
v_out = (1.0-v.T)/s
# Convert v_out to numpy array and save to a .dat file
np.savetxt('v_out.dat', v_out)
print(v_out)
np.savetxt('row_m_np.dat', row_m)
