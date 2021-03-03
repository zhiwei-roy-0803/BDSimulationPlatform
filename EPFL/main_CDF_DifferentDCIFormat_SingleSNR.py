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
A = 16
pesedoA = 57
numCRC = 24
crcPoly = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]
K = A + numCRC
PesodoK = pesedoA + numCRC
SNRdB = 0
SNR = 10**(SNRdB/10)
sigma = np.sqrt(1/SNR)

experiment_name = "CDF-N={:d}-K=[{:d}, {:d}]-SNR={:.1f} dB".format(N, A, pesedoA, SNRdB)
if os.path.isdir(os.path.join(os.path.pardir, "BD_DMetric_EPFL", experiment_name)):
    shutil.rmtree(os.path.join(os.path.pardir, "BD_DMetric_EPFL", experiment_name))
tracer = Tracer('../BD_DMetric_EPFL').attach(experiment_name)

codeConstructor = PolarCodeConstructor(N, K, QPath=os.path.join(os.getcwd(), "../reliable_sequence.txt"))
frozenbits, messagebits, frozenbits_indicator, messagebits_indicator = codeConstructor.PW(N, K)
polarEncoder = PolarEnc(N, K, frozenbits, messagebits)
nodeIdentifier = NodeIdentifier(N, K, frozenbits, messagebits)
nodeType = nodeIdentifier.run()
DMetricCalculator = EPFLDMetricCalculator(N, K, nodeType, frozenbits, messagebits)

frozenbitsPesedo, messagebitsPesedo, _, _ = codeConstructor.PW(N, PesodoK)
polarPesedoEncoder = PolarEnc(N, PesodoK, frozenbitsPesedo, messagebitsPesedo)

crcEnc = CRCEnc(crc_n=numCRC, crc_polynominal=crcPoly)


# Try Complete Noise / Random Codewords / Regular Polar Codes
numSimulation = 2*10**4
DMetricsAnother = []
DMetricsRegular = []
pbar = tqdm(range(numSimulation))
for _ in pbar:
    choice = np.random.randint(low=0, high=2)
    if choice == 0:
        # two mode: pure random codeword / polar encoded but with different information bit length
        msg = np.random.randint(low=0, high=2, size=pesedoA)
        msg_crc = crcEnc.encode(msg)
        cword = polarPesedoEncoder.encode(msg)
        cword = cword.astype(np.int)
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
        DMetricsAnother.append(DMetric)
    else:
        DMetricsRegular.append(DMetric)
DMetricsAnother = np.array(DMetricsAnother)
DMetricsRegular = np.array(DMetricsRegular)

# Plot DMetric CDF for Three Different Scenario
plt.figure(dpi=300)
hist,bins = np.histogram(DMetricsRegular, bins=50, density=True)
cdf = np.cumsum(hist*(bins[1] - bins[0]))
plt.plot(bins[:-1], cdf, color='r', linestyle='-', linewidth=2)

hist,bins = np.histogram(DMetricsAnother, bins=50, density=True)
cdf = np.cumsum(hist*(bins[1] - bins[0]))
plt.plot(bins[:-1], cdf, color='k', linestyle='-', linewidth=2)


plt.xlabel("DMetric")
plt.ylabel("CDF")
plt.legend(["Regular", "Another"])
plt.grid()
tracer.store(plt.gcf(), "CDF.png")