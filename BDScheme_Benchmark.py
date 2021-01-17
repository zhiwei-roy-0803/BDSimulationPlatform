import numpy as np
from genenerate_PDCCH_candidate import CandidateGenerator
from PolarDecoder.Decoder.SCLDecoder import SCLDecoder
from PolarBDEnc.Encoder.CRCEnc import CRCEnc
from tqdm import tqdm
from torchtracer import Tracer
from torchtracer.data import Config
import argparse
import matplotlib.pyplot as plt
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, default=8)
args = parser.parse_args()

# Simulation Parameters
L = args.L

# simulation parameter configuration
numSimulation = 10**4
SNRdBTest = [3]

# initialize Trace instance to maintain simulation hyper-parameters and results and images
experiment_name = "L = {:d}".format(L)
if os.path.isdir(os.path.join(os.getcwd(), "BD_Benchmark", experiment_name)):
    shutil.rmtree(os.path.join(os.getcwd(), "BD_Benchmark", experiment_name))
tracer = Tracer('BD_Benchmark').attach(experiment_name)
configure = {"L": L,
             "numSimulation":numSimulation,
             "SNRdBTest":SNRdBTest}
tracer.store(Config(configure))

# Initialize PDCCH candidate generator and corresponding decoders
PDCCHGenerator = CandidateGenerator()
CRCEncoder = CRCEnc(PDCCHGenerator.numCRCBits, PDCCHGenerator.crcPoly)
SCLDecoders = []
for i in range(PDCCHGenerator.numAggregationLevel):
    N = PDCCHGenerator.codewordLengthPerAggregationLevel[i]
    for m in range(2):
        K = PDCCHGenerator.numInformationBits[m] +  PDCCHGenerator.numCRCBits
        frozenbits_indicator = PDCCHGenerator.frozenbits_indicator_set[i][m]
        messagebits_indicator = PDCCHGenerator.messagebits_indicator_set[i][m]
        SCLDec = SCLDecoder(N=N, K=K, L=L, frozen_bits=frozenbits_indicator, message_bits=messagebits_indicator)
        SCLDecoders.append(SCLDec)

# Start Simulation MDR
print("Simulation for MDR")
MDR_Stage_SNR = []
for SNRdB in SNRdBTest:
    SNR = 10**(SNRdB/10)  # linear scale snr
    sigma = np.sqrt(1/SNR)  # Gaussian noise variance for current EbN0
    pbar = tqdm(range(numSimulation))
    numMissDetection = 0
    numRun = 0
    # Start Simulation for MDR in Current Eb/N0
    for _ in pbar:
        information_bits, codewords, RNTI, RNTIIndex = PDCCHGenerator.generate_candidates(isRNTI=True)
        passedIndex = []
        passedPMs = []
        cnt = 0
        # brute force SC decoding for each candidate
        for i in range(PDCCHGenerator.numAggregationLevel):
            for j in range(2):
                dec_idx = i * 2 + j
                for m in range(PDCCHGenerator.numCandidatePerAggregationLevel[i]):
                    cword = codewords[cnt]
                    cword = cword.astype(np.int)
                    bpsksymbols = 1 - 2 * cword
                    receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, len(cword)))
                    receive_symbols_llr = receive_symbols * (2/sigma**2)
                    decoded_bits, PM = SCLDecoders[dec_idx].decode(receive_symbols_llr)
                    dec_information_bits = decoded_bits[:-PDCCHGenerator.numCRCBits]
                    dec_crc = decoded_bits[-PDCCHGenerator.numCRCBits:]
                    crcCheck = CRCEncoder.encode(dec_information_bits)[-PDCCHGenerator.numCRCBits:]
                    crcCheck[-PDCCHGenerator.numRNTIBits:] = np.mod(crcCheck[-PDCCHGenerator.numRNTIBits:] + RNTI, 2)
                    if np.all(crcCheck == dec_crc):
                        passedIndex.append(cnt)
                        passedPMs.append(PM)
                    cnt += 1

        numPass = len(passedIndex)
        if numPass == 0:
            numMissDetection += 1
        else:
            maxPMIndex = np.argmin(passedPMs)
            finalCandidateIndex = passedIndex[maxPMIndex]
            if finalCandidateIndex != RNTIIndex:
                numMissDetection += 1
        pbar.set_description("Miss Det = {:d}".format(numMissDetection))
        numRun += 1
        if numMissDetection >= 100:
            break

    # Summary Statistic: MDR, FAR
    MDR_Stage = numMissDetection / numRun
    MDR_Stage_SNR.append(MDR_Stage)
    print("SNR = {:.1f} dB, MDR = {:.5f}".format(SNRdB, MDR_Stage))
    tracer.log("{:.6f}".format(MDR_Stage), file="MDR_Stage")


# Plot result for MDR of two stages
plt.figure(dpi=300)
plt.semilogy(SNRdBTest, MDR_Stage_SNR, color='r', linestyle='-', marker="*", markersize=5)
plt.legend(["Stage1", "Stage2"])
plt.xlabel("SNR (dB)")
plt.ylabel("Miss Detection Rate (MDR)")
plt.grid()
tracer.store(plt.gcf(), "MDR.png")

