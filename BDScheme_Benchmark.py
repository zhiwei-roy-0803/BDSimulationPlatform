import numpy as np
from genenerate_PDCCH_candidate import CandidateGenerator
from PolarDecoder.Decoder.CASCLDecoder import CASCLDecoder
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
parser.add_argument("--RUN_FAR", type=int, default=0)
args = parser.parse_args()

# Simulation Parameters
L = args.L
RUN_FAR = args.RUN_FAR
RUN_MDR = 1 - RUN_FAR

# simulation parameter configuration
SNRdBTest = [-3, -2, -1, 0, 1, 2]

# initialize Trace instance to maintain simulation hyper-parameters and results and images
if RUN_MDR == True:
    experiment_name = "MDR-L = {:d}".format(L)
else:
    experiment_name = "FAR-L = {:d}".format(L)
if os.path.isdir(os.path.join(os.getcwd(), "BD_Benchmark", experiment_name)):
    shutil.rmtree(os.path.join(os.getcwd(), "BD_Benchmark", experiment_name))
tracer = Tracer('BD_Benchmark').attach(experiment_name)
configure = {"L": L,
             "SNRdBTest":SNRdBTest}
tracer.store(Config(configure))

# Initialize PDCCH candidate generator and corresponding decoders
PDCCHGenerator = CandidateGenerator()
CRCEncoder = CRCEnc(PDCCHGenerator.numCRCBits, PDCCHGenerator.crcPoly)
CASCLDecoders = []
for i in range(PDCCHGenerator.numAggregationLevel):
    N = PDCCHGenerator.codewordLengthPerAggregationLevel[i]
    for m in range(2):
        A = PDCCHGenerator.numInformationBits[m]
        K = A +  PDCCHGenerator.numCRCBits
        frozenbits_indicator = PDCCHGenerator.frozenbits_indicator_set[i][m]
        messagebits_indicator = PDCCHGenerator.messagebits_indicator_set[i][m]
        SCLDec = CASCLDecoder(N=N, K=K, A=A, L=L, frozen_bits=frozenbits_indicator, message_bits=messagebits_indicator,
                              crc_n=PDCCHGenerator.numCRCBits, crc_p=PDCCHGenerator.crcPoly)
        CASCLDecoders.append(SCLDec)

if RUN_MDR:
    # Start Simulation MDR
    print("Simulation for MDR")
    numSimulation = 10 ** 4
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
                        decoded_bits, PM, isPass = CASCLDecoders[dec_idx].decode(receive_symbols_llr, RNTI)
                        if isPass:
                            passedIndex.append(cnt)
                            passedPMs.append(PM)
                        cnt += 1
            #
            numPass = len(passedIndex)
            if numPass == 0:
                numMissDetection += 1
            else:
                maxPMIndex = int(np.argmin(passedPMs))
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
    plt.legend(["MDR"])
    plt.xlabel("SNR (dB)")
    plt.ylabel("Miss Detection Rate (MDR)")
    plt.grid()
    tracer.store(plt.gcf(), "MDR.png")

if RUN_FAR:
    # Start Simulation MDR
    print("Simulation for FAR")
    numSimulation = 10**6
    FAR_SNR = []
    for SNRdB in SNRdBTest:
        SNR = 10 ** (SNRdB / 10)  # linear scale snr
        sigma = np.sqrt(1 / SNR)  # Gaussian noise variance for current EbN0
        pbar = tqdm(range(numSimulation))
        numFalseAlarm = 0
        numRun = 0
        # Start Simulation for MDR in Current Eb/N0
        for _ in pbar:
            information_bits, codewords, RNTI, RNTIIndex = PDCCHGenerator.generate_candidates(isRNTI=False)
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
                        decoded_bits, PM, isPass = CASCLDecoders[dec_idx].decode(receive_symbols_llr, RNTI)
                        if isPass:
                            passedIndex.append(cnt)
                            passedPMs.append(PM)
                        cnt += 1
            #
            numPass = len(passedIndex)
            if numPass > 0:
                numFalseAlarm += 1

            pbar.set_description("Miss Det = {:d}".format(numFalseAlarm))
            numRun += 1
            if numFalseAlarm >= 10:
                break

        # Summary Statistic: MDR, FAR
        FAR = numFalseAlarm / numRun
        FAR_SNR.append(FAR)
        print("SNR = {:.1f} dB, FAR = {:.5f}".format(SNRdB, FAR))
        tracer.log("{:.6f}".format(FAR), file="FAR")

    # Plot result for MDR of two stages
    plt.figure(dpi=300)
    plt.semilogy(SNRdBTest, FAR_SNR, color='r', linestyle='-', marker="*", markersize=5)
    plt.legend(["FAR"])
    plt.xlabel("SNR (dB)")
    plt.ylabel("False Alarm Rate (FAR)")
    plt.grid()
    tracer.store(plt.gcf(), "FAR.png")