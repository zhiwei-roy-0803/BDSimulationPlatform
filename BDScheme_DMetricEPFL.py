import numpy as np
from DMetricCalulator.EPFLDMetric import EPFLDMetricCalculator
from utils.IdentifyNodes import NodeIdentifier
from utils.CodeConstruction import PolarCodeConstructor
import os
from PolarBDEnc.Encoder.CRCEnc import CRCEnc
from PolarBDEnc.Encoder.PolarEnc import PolarEnc
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchtracer import Tracer
import shutil

N = 512
A = 57
numCRC = 16
K = A + numCRC
PesodoK = 32 + numCRC
rate = A/N
SNRdB = 3
SNR = 10**(SNRdB/10)
sigma = np.sqrt(1/SNR)

experiment_name = "CDF-N={:d}-K={:d}-SNR={:.1f}".format(N, K, SNR)
if os.path.isdir(os.path.join(os.getcwd(), "BD_DMetric_EPFL", experiment_name)):
    shutil.rmtree(os.path.join(os.getcwd(), "BD_DMetric_EPFL", experiment_name))
tracer = Tracer('BD_DMetric_EPFL').attach(experiment_name)

codeConstructor = PolarCodeConstructor(N, K, QPath=os.path.join(os.getcwd(), "reliable_sequence.txt"))
frozenbits, messagebits, frozenbits_indicator, messagebits_indicator = codeConstructor.PW(N, K)
polarEncoder = PolarEnc(N, K, frozenbits, messagebits)
nodeIdentifier = NodeIdentifier(N, K, frozenbits, messagebits)
nodeType = nodeIdentifier.run()
DMetricCalculator = EPFLDMetricCalculator(N, K, nodeType, frozenbits, messagebits)

frozenbitsPesedo, messagebitsPesedo, _, _ = codeConstructor.PW(N, PesodoK)
polarPesedoEncoder = PolarEnc(N, PesodoK, frozenbitsPesedo, messagebitsPesedo)

crcEnc = CRCEnc(crc_n=numCRC, crc_polynominal=[16, 15, 2, 0])


# Try Complete Noise / Random Codewords / Regular Polar Codes
numSimulation = 3*10**4
DMetricsRandom = []
DMetricsRegular = []
DMetricsNoise = []
pbar = tqdm(range(numSimulation))
for _ in pbar:
    choice = np.random.randint(low=0, high=3)
    # choice = 2
    if choice == 0:
        bpsksymbols = np.zeros(N, dtype=int)
    elif choice == 1:
        # two mode: pure random codeword / polar encoded but with different information bit length
        cword = np.random.randint(low=0, high=2, size=N, dtype=int)
        # msg = np.random.randint(low=0, high=2, size=PesodoK)
        # cword = polarPesedoEncoder.encode(msg)
        bpsksymbols = 1 - 2 * cword
    else:
        msg = np.random.randint(low=0, high=2, size=A)
        msg_crc = crcEnc.encode(msg)
        cword = polarEncoder.encode(msg_crc)
        cword = cword.astype(np.int)
        bpsksymbols = 1 - 2 * cword

    receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))
    receive_symbols_llr = receive_symbols * (2 / sigma ** 2)
    DMetric = DMetricCalculator.getDMetric(receive_symbols_llr)
    if choice == 0:
        DMetricsNoise.append(DMetric)
    elif choice == 1:
        DMetricsRandom.append(DMetric)
    else:
        DMetricsRegular.append(DMetric)
DMetricsNoise = np.array(DMetricsNoise)
DMetricsRandom = np.array(DMetricsRandom)
DMetricsRegular = np.array(DMetricsRegular)

# Plot DMetric CDF for Three Different Scenario
plt.figure(dpi=300)
hist,bins = np.histogram(DMetricsNoise, bins=50, density=True)
cdf = np.cumsum(hist*(bins[1] - bins[0]))
plt.plot(bins[:-1], cdf, color='r', linestyle='-', linewidth=2)

hist,bins = np.histogram(DMetricsRandom, bins=50, density=True)
cdf = np.cumsum(hist*(bins[1] - bins[0]))
plt.plot(bins[:-1], cdf, color='k', linestyle='-', linewidth=2)

hist,bins = np.histogram(DMetricsRegular, bins=50, density=True)
cdf = np.cumsum(hist*(bins[1] - bins[0]))
plt.plot(bins[:-1], cdf, color='b', linestyle='-', linewidth=2)

plt.xlabel("DMetric")
plt.ylabel("CDF")
plt.legend(["Complete Noise", "Random Codeword", "Regular Codeword"])
plt.grid()
tracer.store(plt.gcf(), "CDF.png")

# Try Pesudo and Regular
DMetricsPesudo = []
DMetricsRegular = []
numSimulation = 2 * 10**4
pbar = tqdm(range(numSimulation))
for _ in pbar:
    choice = np.random.randint(low=0, high=2)
    if choice == 0:
        # polar encoded but with different information bit length
        msg = np.random.randint(low=0, high=2, size=PesodoK)
        cword = polarPesedoEncoder.encode(msg)
        bpsksymbols = 1 - 2 * cword
    else:
        msg = np.random.randint(low=0, high=2, size=A)
        msg_crc = crcEnc.encode(msg)
        cword = polarEncoder.encode(msg_crc)
        cword = cword.astype(np.int)
        bpsksymbols = 1 - 2 * cword

    receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))
    receive_symbols_llr = receive_symbols * (2 / sigma ** 2)
    DMetric = DMetricCalculator.getDMetric(receive_symbols_llr)
    if choice == 0:
        DMetricsPesudo.append(DMetric)
    else:
        DMetricsRegular.append(DMetric)

DMetricsRandom = np.array(DMetricsRandom)
DMetricsRegular = np.array(DMetricsRegular)


# Plot DMetric CDF for Two Different Scenarios
plt.figure(dpi=300)
hist,bins = np.histogram(DMetricsPesudo, bins=50, density=True)
cdf = np.cumsum(hist*(bins[1] - bins[0]))
plt.plot(bins[:-1], cdf, color='b', linestyle='-', linewidth=2)
plt.xlabel("DMetric")
plt.ylabel("CDF")
plt.legend(["Pesudo Polar Codes ({:d}, {:d})".format(N, PesodoK)])
plt.grid()
tracer.store(plt.gcf(), "CDF_PesudoPolar.png")

hist,bins = np.histogram(DMetricsRegular, bins=50, density=True)
cdf = np.cumsum(hist*(bins[1] - bins[0]))
plt.plot(bins[:-1], cdf, color='b', linestyle='-', linewidth=2)
plt.xlabel("DMetric")
plt.ylabel("CDF")
plt.legend(["Polar Codes ({:d}, {:d})".format(N, K)])
plt.grid()
tracer.store(plt.gcf(), "CDF_Polar.png")