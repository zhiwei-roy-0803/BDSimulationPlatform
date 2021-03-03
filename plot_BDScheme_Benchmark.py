import numpy as np
import matplotlib.pyplot as plt

MDR_BDScheme_1 = np.loadtxt("/Users/zhiweicao/Desktop/Low Latency Polar Decoder/BlindDetection/BDSimulationPlatform/BD_Kuangda/MDR-L1=1-L2=8-Omega=11/MDR.log")
MDR_BDScheme_2 = np.loadtxt("/Users/zhiweicao/Desktop/Low Latency Polar Decoder/BlindDetection/BDSimulationPlatform/BD_Kuangda/MDR-L1=2-L2=8-Omega=11/MDR.log")
MDR_BDScheme_4 = np.loadtxt("/Users/zhiweicao/Desktop/Low Latency Polar Decoder/BlindDetection/BDSimulationPlatform/BD_Kuangda/MDR-L1=4-L2=8-Omega=11/MDR.log")
MDR_Benchmark = np.loadtxt("/Users/zhiweicao/Desktop/Low Latency Polar Decoder/BlindDetection/BDSimulationPlatform/BD_Benchmark/L = 8/MDR_Stage.log")

SNRdBTest = [0, 1, 2, 3, 4]
numSNR = len(SNRdBTest)
plt.figure(dpi=300)
plt.semilogy(SNRdBTest, MDR_BDScheme_1[:numSNR], color='b', linestyle='-', marker="o", markersize=5)
plt.semilogy(SNRdBTest, MDR_BDScheme_2[:numSNR], color='r', linestyle='-', marker="o", markersize=5)
plt.semilogy(SNRdBTest, MDR_BDScheme_4[:numSNR], color='g', linestyle='-', marker="o", markersize=5)
plt.semilogy(SNRdBTest, MDR_Benchmark[:numSNR], color='k', linestyle='-', marker="o", markersize=5)
plt.legend(["KuangDa-L1=1", "KuangDa-L1=2", "KuangDa-L1=4", "Benchmark-L=8"])
plt.xlabel("SNR (dB)")
plt.ylabel("Miss Detection Rate (MDR)")
plt.grid()
plt.show()