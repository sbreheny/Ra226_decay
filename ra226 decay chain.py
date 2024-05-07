#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 03:12:52 2019

@author: sbreheny
"""

# Simulate radioactive decay of Ra226

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import math

Na=6.022e23
# Each of these returns a list of tuples
# In each tuple, the first entry is the particle energy in keV
# The second entry is the emission rate per second
# The third entry is the type

# Cluster decay and spontaneous fission are ignored

# Gamma and X-Ray
# For continuous spectrum emissions, the median energy is given
# Types: 
# X - X-ray (from electron cloud)
# G - Gamma (from nucleus)
def EM_Output(state_vector):
    out=[]
    # From Ra-226
    dps=(Na/226.025)*state_vector[0]/half_life_to_tc(years_to_seconds(1600.0))
    out.append(((10.14+17.26)/2.0,0.00807*dps,'X'))
    out.append((81.07,0.00192*dps,'X'))
    out.append((83.78,0.00317*dps,'X'))
    out.append((95.0,0.001098*dps,'X'))
    out.append((98.0,0.000351*dps,'X'))
    out.append((186.211,0.03555*dps,'G'))
    # From Rn-222
    dps=(Na/222.0)*state_vector[1]/half_life_to_tc(days_to_seconds(3.82))
    out.append((510.0,0.00076*dps,'G'))
    # From Po-218
    dps=(Na/218.0)*state_vector[2]/half_life_to_tc(60.0*3.1)
    out.append((836.0,0.000011*dps,'G'))
    # From Pb-214
    dps=(Na/214.0)*state_vector[3]/half_life_to_tc(60.0*26.8)
    out.append(((9.42+16.36)/2.0,0.1242*dps,'X'))
    out.append((74.8157,0.0626*dps,'X'))
    out.append((77.1088,0.1047*dps,'X'))
    out.append((241.997,0.07268*dps,'G'))
    out.append((295.224,0.18414*dps,'G'))
    out.append((351.932,0.356*dps,'G'))
    # From At-218
    # None
    # From Bi-214
    dps=(Na/214.0)*state_vector[5]/half_life_to_tc(60.0*19.9)
    out.append((609.312,0.4549*dps,'G'))
    out.append((768.356,0.04892*dps,'G'))
    out.append((1120.287,0.1491*dps,'G'))
    out.append((1238.111,0.05381*dps,'G'))
    out.append((1377.669,0.03968*dps,'G'))
    out.append((1764.494,0.1531*dps,'G'))
    out.append((2204.21,0.04913*dps,'G'))
    # From Rn-218
    dps=(Na/218.0)*state_vector[6]/half_life_to_tc(0.035)
    out.append((609.31,0.00124*dps,'G'))
    # From Po-214
    # Negligible
    # From Tl-210
    dps=(Na/210.0)*state_vector[8]/half_life_to_tc(60.0*1.3)
    out.append((72.805,0.07*dps,'X'))
    out.append((74.97,0.11*dps,'X'))
    out.append(((84.451+85.47)/2.0,0.038*dps,'X'))
    out.append(((87.238+87.911)/2.0,0.011*dps,'X'))
    out.append((296.0,0.79*dps,'G'))
    out.append((799.6,0.98969*dps,'G'))
    out.append((1210.0,0.168*dps,'G'))
    out.append((1310.0,0.208*dps,'G'))
    # From Pb-210
    dps=(Na/210.0)*state_vector[9]/half_life_to_tc(years_to_seconds(22.2))
    out.append(((9.4207+15.7084)/2.0,0.22*dps,'X'))
    out.append((46.539,0.04252*dps,'G'))
    # From Bi-210
    # Negligible
    # From Hg-206
    dps=(Na/206.0)*state_vector[11]/half_life_to_tc(60.0*8.15)
    out.append(((8.9531+14.7362)/2.0,0.029*dps,'X'))
    out.append((70.8325,0.023*dps,'X'))
    out.append((72.8725,0.039*dps,'X'))
    out.append(((82.118+83.115)/2.0,0.0132*dps,'X'))
    out.append((304.896,0.26*dps,'G'))
    out.append((649.42,0.022*dps,'G'))
    # From Po-210
    # Negligible
    # From Tl-206
    # Negligible
    return out


# Alphas
# Type is always A
def Alpha_Output(state_vector):
    out=[]
    # From Ra-226
    dps=(Na/226.0)*state_vector[0]/half_life_to_tc(years_to_seconds(1600.0))
    out.append((4601.0,0.0595*dps,'A'))
    out.append((4784.34,0.94038*dps,'A'))
    # From Rn-222
    dps=(Na/222.0)*state_vector[1]/half_life_to_tc(days_to_seconds(3.82))
    out.append((5489.48,0.9992*dps,'A'))
    # From Po-218
    dps=(Na/218.0)*state_vector[2]/half_life_to_tc(60.0*3.1)
    out.append((6002.35,0.999769*dps,'A'))
    # From Pb-214
    # None
    # From At-218
    dps=(Na/218.0)*state_vector[4]/half_life_to_tc(1.5)
    out.append((6653.0,0.064*dps,'A'))
    out.append((6694.0,0.90*dps,'A'))
    out.append((6756.0,0.036*dps,'A'))
    # From Bi-214
    # Negligible
    # From Rn-218
    dps=(Na/218.0)*state_vector[6]/half_life_to_tc(0.035)
    out.append((6531.1,0.00127*dps,'A'))
    out.append((7129.2,0.99873*dps,'A'))
    # From Po-214
    dps=(Na/214.0)*state_vector[7]/half_life_to_tc(1.643e-4)
    out.append((7686.82,0.999895*dps,'A'))
    # From Tl-210
    # None
    # From Pb-210
    # Negligible
    # From Bi-210
    # Negligible
    # From Hg-206
    # None
    # From Po-210
    dps=(Na/210.0)*state_vector[12]/half_life_to_tc(days_to_seconds(138.376))
    out.append((5304.33,0.9999876*dps,'A'))
    # From Tl-206
    # Negligible
    return out

# Charged particles (electron and positrons from beta decay,
# electron capture, and Auger effect)
# Types:
# B+ - Beta positron
# B- - Beta electron
# EC - Electron Capture
# U - Auger electron
# For continuous spectrum emissions, the median energy is given
# For beta, the average energy is given
def CP_Output(state_vector):
    out=[]
    # From Ra-226
    dps=(Na/226.0)*state_vector[0]/half_life_to_tc(years_to_seconds(1600.0))
    out.append(((87.814+186.168)/2.0,0.02407*dps,'EC'))
    out.append((87.814,0.00675*dps,'EC'))
    out.append(((168.163+171.600)/2.0,0.0128*dps,'EC'))
    out.append(((181.738+183.327)/2.0,0.00342*dps,'EC'))
    # From Rn-222
    # None
    # From Po-218
    dps=(Na/218.0)*state_vector[2]/half_life_to_tc(60.0*3.1)
    out.append((73.0,0.00022*dps,'B-'))
    # From Pb-214
    dps=(Na/214.0)*state_vector[3]/half_life_to_tc(60.0*26.8)
    out.append(((5.3+16.4)/2.0,0.198*dps,'U'))
    out.append(((57.49+90.52)/2.0,0.008*dps,'U'))
    out.append(((36.8+39.8)/2.0,0.1039*dps,'EC'))
    out.append(((49.2284+50.6479)/2.0,0.0246*dps,'EC'))
    out.append((151.471,0.0526*dps,'EC'))
    out.append((204.698,0.0722*dps,'EC'))
    out.append((261.406,0.0926*dps,'EC'))
    out.append(((278.836+281.805)/2.0,0.01291*dps,'EC'))
    out.append(((335.544+338.513)/2.0,0.01584*dps,'EC'))
    out.append((50.0,0.02762*dps,'B-'))
    out.append((145.0,0.01047*dps,'B-'))
    out.append((207.0,0.4652*dps,'B-'))
    out.append((227.0,0.4109*dps,'B-'))
    out.append((337.0,0.092*dps,'B-'))
    # From At-218
    dps=(Na/218.0)*state_vector[4]/half_life_to_tc(1.5)
    out.append((1095.0,0.001*dps,'B-'))
    # From Bi-214
    dps=(Na/214.0)*state_vector[5]/half_life_to_tc(60.0*19.9)
    out.append((493.0,0.08147*dps,'B-'))
    out.append((526.0,0.1710*dps,'B-'))
    out.append((540.0,0.17494*dps,'B-'))
    out.append((1270.0,0.1967*dps,'B-'))
    # From Rn-218
    # None
    # From Po-214
    # None
    # From Tl-210
    dps=(Na/210.0)*state_vector[8]/half_life_to_tc(60.0*1.3)
    out.append((9.0,0.16*dps,'EC'))
    out.append(((67.1392+69.9648)/2.0,0.20*dps,'EC'))
    out.append((674.0,0.24*dps,'B-'))
    out.append((1721.0,0.31*dps,'B-'))
    # From Pb-210
    dps=(Na/210.0)*state_vector[9]/half_life_to_tc(years_to_seconds(22.2))
    out.append(((5.3+10.7)/2.0,0.360*dps,'U'))
    out.append(((42.540+43.959)/2.0,0.1365*dps,'EC'))
    out.append((4.3,0.802*dps,'B-'))
    out.append((16.3,0.198*dps,'B-'))
    # From Bi-210
    dps=(Na/210.0)*state_vector[10]/half_life_to_tc(days_to_seconds(5.013))
    out.append((317.0,0.9999986*dps,'B-'))
     # From Hg-206
    dps=(Na/206.0)*state_vector[11]/half_life_to_tc(60.0*8.15)
    out.append(((5.25+15.32)/2.0,0.051*dps,'U'))
    out.append((219.366,0.08*dps,'EC'))
    out.append((203.0,0.03*dps,'B-'))
    out.append((330.0,0.35*dps,'B-'))
    out.append((450.0,0.62*dps,'B-'))
    # From Po-210
    # Negligible
    # From Tl-206
    dps=(Na/206.00)*state_vector[13]/half_life_to_tc(60.0*4.2)
    out.append((538.86,0.99885*dps,'B-'))
    return out

def rate_from_tuples(in_list):
    return math.fsum([c[1] for c in in_list])

def rate_trajectory(fctn,my_matrix):
    out=[]
    for index in range(Npoints):
        out.append(rate_from_tuples(fctn(my_matrix[:,index])))
    return np.array(out)


def spectrum_from_tuples(in_list,abs_tol):
    nrg=[]
    rate=[]
    for c in in_list:
        found_flag=False
        for index,value in enumerate(nrg):
            if value-abs_tol <= c[0] <= value+abs_tol:
                rate[index]+=c[1]
                found_flag=True
                break
        if not found_flag:
            nrg.append(c[0])
            rate.append(c[1])
    nrg=np.array(nrg)
    rate=np.array(rate)
    indices=np.argsort(nrg)
    nrg=nrg[indices]
    rate=rate[indices]
    return (nrg,rate)

               

def half_life_to_tc(hl):
    return hl/np.log(2.0)

def years_to_seconds(yr):
    return yr*365.2422*24.0*3600.0

def days_to_seconds(dy):
    return 24.0*3600.0*dy

# State vector, quantities in grams
# 0, Ra-226, 226.025, 1600y
# 1, Rn-222, 222.018, 3.82d
# 2, Po-218, 218.01, 3.1m
# 3, Pb-214, 214.00, 26.8m
# 4, At-218, 218.01, 1.5s
# 5, Bi-214, 214.00, 19.9m
# 6, Rn-218, 218.01, 3.5e-2s
# 7, Po-214, 214.00, 1.643e-4s
# 8, Tl-210, 209.99, 1.30m
# 9, Pb-210, 209.98, 22.20y
# 10, Bi-210, 209.98, 5.013d
# 11, Hg-206, 205.978, 8.15m
# 12, Po-210, 209.98, 138.376d
# 13, Tl-206, 205.976, 4.200m
# 14, Pb-206 205.974, stable

# Conversions with probabilities and emission type
# 0->1 alpha
# 1->2 alpha
# 2->3 1-2e-4 alpha, 2->4 2e-4 beta-
# 3->5 beta-
# 4->5 0.999 alpha, 4->6 0.001 beta-
# 5->7 1-2.1e-4 beta-, 5->8 2.1e-4 alpha
# 6->7 alpha
# 7->9 alpha
# 8->9 beta-
# 9->10 1-1.9e-8 beta-, 9->11 1.9e-8 alpha
# 10->12 1-1.32e-6 beta-, 10->13 1.32e-6 alpha
# 11->13 beta-
# 12->14 alpha
# 13->14 beta-

Npoints=1000
tfinal=years_to_seconds(0.05)
dt=tfinal/Npoints
conv_mat=np.zeros((15,15))
msv=np.zeros((15,Npoints))
msv[0,0]=3e-8 # Initial quantity of Ra-226


# Pure decay terms
conv_mat[0,0]=-1.0/half_life_to_tc(years_to_seconds(1600.0))
conv_mat[1,1]=-1.0/half_life_to_tc(days_to_seconds(3.82))
conv_mat[2,2]=-1.0/half_life_to_tc(60.0*3.1)
conv_mat[3,3]=-1.0/half_life_to_tc(60.0*26.8)
conv_mat[4,4]=-1.0/half_life_to_tc(1.5)
conv_mat[5,5]=-1.0/half_life_to_tc(60.0*19.9)
conv_mat[6,6]=-1.0/half_life_to_tc(0.035)
conv_mat[7,7]=-1.0/half_life_to_tc(1.643e-4)
conv_mat[8,8]=-1.0/half_life_to_tc(60.0*1.3)
conv_mat[9,9]=-1.0/half_life_to_tc(years_to_seconds(22.2))
conv_mat[10,10]=-1.0/half_life_to_tc(days_to_seconds(5.013))
conv_mat[11,11]=-1.0/half_life_to_tc(60.0*8.15)
conv_mat[12,12]=-1.0/half_life_to_tc(days_to_seconds(138.376))
conv_mat[13,13]=-1.0/half_life_to_tc(60.0*4.2)

# Cross-terms
# [contributes to, comes from]
conv_mat[1,0]=(222.018/226.025)/half_life_to_tc(years_to_seconds(1600.0))
conv_mat[2,1]=(218.00/222.018)/half_life_to_tc(days_to_seconds(3.82))
conv_mat[3,2]=(1.0-2e-4)*(214.00/218.00)/half_life_to_tc(60.0*3.1)
conv_mat[4,2]=(2e-4)*(218.01/218.01)/half_life_to_tc(60.0*3.1)
conv_mat[5,3]=(214.00/214.00)/half_life_to_tc(60.0*26.8)
conv_mat[5,4]=0.999*(214.00/218.01)/half_life_to_tc(1.5)
conv_mat[6,4]=0.001*(218.01/218.01)/half_life_to_tc(1.5)
conv_mat[7,5]=(1.0-2.1e-4)*(214.00/214.00)/half_life_to_tc(60.0*19.9)
conv_mat[7,6]=(214.00/218.01)/half_life_to_tc(0.035)
conv_mat[8,5]=(2.1e-4)*(209.99/214.00)/half_life_to_tc(60.0*19.9)
conv_mat[9,7]=(209.98/214.00)/half_life_to_tc(1.643e-4)
conv_mat[9,8]=(209.98/209.99)/half_life_to_tc(60.0*1.3)
conv_mat[10,9]=(1.0-1.9e-8)*(209.98/209.98)/half_life_to_tc(years_to_seconds(22.2))
conv_mat[11,9]=(1.9e-8)*(205.978/209.98)/half_life_to_tc(years_to_seconds(22.2))
conv_mat[12,10]=(1.0-1.32e-6)*(209.98/209.98)/half_life_to_tc(days_to_seconds(5.013))
conv_mat[13,10]=(1.32e-6)*(205.976/209.98)/half_life_to_tc(days_to_seconds(5.013))
conv_mat[13,11]=(205.976/205.978)/half_life_to_tc(60.0*8.15)
conv_mat[14,12]=(205.974/209.98)/half_life_to_tc(days_to_seconds(138.376))
conv_mat[14,13]=(205.974/205.976)/half_life_to_tc(60.0*4.2)

# Time in seconds, for 100 years
tvect=np.linspace(0,tfinal,num=Npoints)

for index,t in enumerate(tvect):
    if index>0:
        msv[:,index]=np.matmul(sp.linalg.expm(conv_mat*dt),msv[:,index-1])

plt.semilogy(tvect,msv[0,:],label='0:Ra-226')
plt.semilogy(tvect,msv[1,:],label='1:Rn-222')
plt.semilogy(tvect,msv[2,:],label='2:Po-218')
plt.semilogy(tvect,msv[3,:],label='3:Pb-214')
plt.semilogy(tvect,msv[4,:],label='4:At-218')
plt.semilogy(tvect,msv[5,:],label='5:Bi-214')
plt.semilogy(tvect,msv[6,:],label='6:Rn-218')
plt.semilogy(tvect,msv[7,:],label='7:Po-214')
plt.semilogy(tvect,msv[8,:],label='8:Tl-210')
plt.semilogy(tvect,msv[9,:],label='9:Pb-210')
plt.semilogy(tvect,msv[10,:],label='10:Bi-210')
plt.semilogy(tvect,msv[11,:],label='11:Hg-206')
plt.semilogy(tvect,msv[12,:],label='12:Po-210')
plt.semilogy(tvect,msv[13,:],label='13:Tl-206')
plt.semilogy(tvect,msv[14,:],label='14:Pb-206')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Quantity (grams)')
plt.title('Ra-226 Decay')

plt.figure(2)
plt.plot(tvect,rate_trajectory(EM_Output,msv),label='Gamma + X-Ray')
plt.plot(tvect,rate_trajectory(CP_Output,msv),label='Beta, EC, and Auger')
plt.plot(tvect,rate_trajectory(Alpha_Output,msv),label='Alpha')
plt.xlabel('Time (s)')
plt.ylabel('Rate (events/sec)')
plt.title('Radium 226 Radiation Time-history')
plt.legend()

plt.figure(3)
(a,b)=spectrum_from_tuples(EM_Output(msv[:,Npoints-1]),5.0)
plt.stem(a,b)
plt.xlabel('Energy (keV)')
plt.ylabel('Rate (events/sec)')
plt.title('Ra-226 and Daughters Gamma Spectrum')

plt.figure(4)
(a,b)=spectrum_from_tuples(CP_Output(msv[:,Npoints-1]),5.0)
plt.stem(a,b)
plt.xlabel('Energy (keV)')
plt.ylabel('Rate (events/sec)')
plt.title('Ra-226 and Daughters Beta/EC/Auger Spectrum')

plt.figure(5)
(a,b)=spectrum_from_tuples(Alpha_Output(msv[:,Npoints-1]),5.0)
plt.stem(a,b)
plt.xlabel('Energy (keV)')
plt.ylabel('Rate (events/sec)')
plt.title('Ra-226 and Daughters Alpha Spectrum')


if tvect[-1]>years_to_seconds(10.0):
    tvect2=tvect/years_to_seconds(1.0)
    plt.figure(6)
    plt.plot(tvect2,rate_trajectory(EM_Output,msv),label='Gamma + X-Ray')
    plt.plot(tvect2,rate_trajectory(CP_Output,msv),label='Beta, EC, and Auger')
    plt.plot(tvect2,rate_trajectory(Alpha_Output,msv),label='Alpha')
    plt.xlabel('Time (years)')
    plt.ylabel('Rate (events/sec)')
    plt.title('Radium 226 Radiation Time-history')
    plt.legend()

