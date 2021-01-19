import numpy as np
from genenerate_PDCCH_candidate import CandidateGenerator
from PolarDecoder.Decoder.SCLDecoder import SCLDecoder
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
parser.add_argument("--Omega", type=int, default=4)
parser.add_argument("--RUN_FAR", type=int, default=0)
args = parser.parse_args()
# Simulation Parameters
Omega = args.Omega # No. candidate after stage 1 SC decoding
L = args.L
RUN_FAR = args.RUN_FAR
RUN_MDR = 1 - RUN_FAR
# simulation parameter configuration
numSimulation = 10**4
SNRdBTest = [-3, -2, -1, 0, 1, 2, 3]
# initialize Trace instance to maintain simulation hyper-parameters and results and images
if RUN_MDR == True:
    experiment_name = "MDR-L={:d} Omega={:d}".format(1, L, Omega)
else:
    experiment_name = "FAR-L={:d} Omega={:d}".format(1, L, Omega)

if os.path.isdir(os.path.join(os.getcwd(), "BD_Gross", experiment_name)):
    shutil.rmtree(os.path.join(os.getcwd(), "BD_Gross", experiment_name))
tracer = Tracer('BD_Gross').attach(experiment_name)
configure = {"L": L,
             "Omega": Omega,
             "numSimulation":numSimulation,
             "SNRdBTest":SNRdBTest}

tracer.store(Config(configure))

# Initialize PDCCH candidate generator and corresponding decoders
PDCCHGenerator = CandidateGenerator()
CRCEncoder = CRCEnc(PDCCHGenerator.numCRCBits, PDCCHGenerator.crcPoly)
SCLOneDecoders = []
CASCLDecoders = []
for i in range(PDCCHGenerator.numAggregationLevel):
    N = PDCCHGenerator.codewordLengthPerAggregationLevel[i]
    for m in range(2):
        A = PDCCHGenerator.numInformationBits[m]
        K = A + PDCCHGenerator.numCRCBits
        frozenbits_indicator = PDCCHGenerator.frozenbits_indicator_set[i][m]
        messagebits_indicator = PDCCHGenerator.messagebits_indicator_set[i][m]
        SCLOneDec = SCLDecoder(N=N, K=K, L=1, frozen_bits=frozenbits_indicator, message_bits=messagebits_indicator)
        CASCLDec = CASCLDecoder(N=N, K=K, A=A, L=L, frozen_bits=frozenbits_indicator, message_bits=messagebits_indicator,
                                crc_n=PDCCHGenerator.numCRCBits, crc_p=PDCCHGenerator.crcPoly)
        SCLOneDecoders.append(SCLOneDec)
        CASCLDecoders.append(CASCLDec)

if RUN_MDR == True:
    # Start Simulation MDR
    print("Simulation for MDR")
    MDR_Stage1_SNR = []
    MDR_Stage2_SNR = []
    for SNRdB in SNRdBTest:
        SNR = 10**(SNRdB/10)  # linear scale snr
        sigma = np.sqrt(1/SNR)  # Gaussian noise variance for current EbN0
        pbar = tqdm(range(numSimulation))
        numMissDetection = 0
        numStage1MissDetection = 0
        numRun = 0
        # Start Simulation for MDR in Current Eb/N0
        for _ in pbar:
            information_bits, codewords, RNTI, RNTIIndex = PDCCHGenerator.generate_candidates(isRNTI=True)
            # -------First stage low complexity SC decoding--------- #
            passedNoisyCodeWord = []
            passedIndex = []
            passedDecIndex = []
            passedMetrics = []

            notPassedNoisyCodeWord = []
            notPassedIndex = []
            notPassedDecIndex = []
            notPassedMetrics = []
            # brute force SC decoding for each candidate
            for i in range(PDCCHGenerator.numAggregationLevel):
                cword_offset = i * PDCCHGenerator.numCandidatePerAggregationLevel[i] * 2
                dec_offset = i * 2
                numCandidateCurrent = PDCCHGenerator.numCandidatePerAggregationLevel[i]
                N = PDCCHGenerator.codewordLengthPerAggregationLevel[i]
                for m in range(numCandidateCurrent):
                    metrics = np.zeros(2)
                    dec_res = []
                    receive_symbols_llrs = []
                    for j in range(2):
                        A = PDCCHGenerator.numInformationBits[j]
                        K = A + PDCCHGenerator.numCRCBits
                        cword = codewords[cword_offset + m + j * numCandidateCurrent]
                        cword = cword.astype(np.int)
                        bpsksymbols = 1 - 2 * cword
                        receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, len(cword)))
                        receive_symbols_llr = receive_symbols * (2/sigma**2)
                        decoded_bits, metric = SCLOneDecoders[dec_offset + j].decode(receive_symbols_llr)
                        metrics[j] = (metric)/(N-K)
                        dec_res.append(decoded_bits)
                        receive_symbols_llrs.append(receive_symbols_llr)
                    # select the candidate with smaller metric value
                    min_idx = np.argmin(metrics)[0]
                    winner = dec_res[min_idx]
                    dec_information_bits = winner[:-PDCCHGenerator.numCRCBits]
                    dec_crc = winner[-PDCCHGenerator.numCRCBits:]
                    crcCheck = CRCEncoder.encode(dec_information_bits)[-PDCCHGenerator.numCRCBits:]
                    crcCheck[-PDCCHGenerator.numRNTIBits:] = crcCheck[-PDCCHGenerator.numRNTIBits:] ^ RNTI
                    if np.all(crcCheck == dec_crc):
                        passedNoisyCodeWord.append(receive_symbols_llrs[min_idx])
                        passedDecIndex.append(dec_offset + min_idx)
                        passedIndex.append(cword_offset + m + min_idx * numCandidateCurrent)
                        passedMetrics.append(metrics[min_idx])
                    else:
                        notPassedNoisyCodeWord.append(receive_symbols_llrs[min_idx])
                        notPassedDecIndex.append(dec_offset + min_idx)
                        notPassedIndex.append(cword_offset + m + min_idx * numCandidateCurrent)
                        notPassedMetrics.append(metrics[min_idx])

            # if there is only one candidate survive, return the decode result as final detection result
            if len(passedMetrics) == 0:
                notPassedMetrics = np.array(notPassedMetrics)
                argIdxNotPassedMetric = np.argsort(notPassedMetrics)[::-1]
                noisyCodeWordStage2 = []
                idxStage2 = []
                idxDecStage2 = []
                for idx in argIdxNotPassedMetric[:Omega]:
                    noisyCodeWordStage2.append(notPassedNoisyCodeWord[idx])
                    idxStage2.append(notPassedIndex[idx])
                    idxDecStage2.append(notPassedDecIndex[idx])
            elif len(passedMetrics) == 1:
                if passedIndex[0] != RNTIIndex:
                    numMissDetection += 1
                    numStage1MissDetection += 1
                pbar.set_description("Miss Det Stage 1 = {:d}, Miss Det Stage 2 = {:d}".format(numStage1MissDetection, numMissDetection))
                continue
            else:
                noisyCodeWordStage2 = passedNoisyCodeWord
                idxStage2 = passedIndex
                idxDecStage2 = passedDecIndex

            # -------Second stage SCL decoding--------- #
            passedIndex = []
            passedPMs = []
            numCandidateStage2 = len(noisyCodeWordStage2)
            for i in range(numCandidateStage2):
                dec_idx = idxDecStage2[i]
                decoded_bits, PM, isPass = CASCLDecoders[dec_idx].decode(noisyCodeWordStage2[i], RNTI)
                if isPass:
                    passedIndex.append(idxStage2[i])
                    passedPMs.append(PM)
            numPass = len(passedIndex)
            if numPass == 0:
                numMissDetection += 1
            else:
                minPMIndexStage2 = np.argmin(passedPMs)
                finalCandidateIndex = passedIndex[minPMIndexStage2]
                if finalCandidateIndex != RNTIIndex:
                    numMissDetection += 1
            pbar.set_description("Miss Det Stage 1 = {:d}, Miss Det Stage 2 = {:d}".format(numStage1MissDetection, numMissDetection))
            numRun += 1
            if numMissDetection >= 300:
                break

        # Summary Statistic: MDR, FAR
        MDR_Stage1 = numStage1MissDetection / numRun
        MDR_Stage2 = numMissDetection / numRun
        MDR_Stage1_SNR.append(MDR_Stage1)
        MDR_Stage2_SNR.append(MDR_Stage2)
        print("SNR = {:.1f} dB, MDR Stage 1 = {:.5f}, MDR Stage 2 = {:.5f}".format(SNRdB, MDR_Stage1, MDR_Stage2))
        tracer.log("{:.6f}".format(MDR_Stage1), file="MDR_Stage1")
        tracer.log("{:.6f}".format(MDR_Stage2), file="MDR_Stage2")

    # Plot result for MDR of two stages
    plt.figure(dpi=300)
    plt.semilogy(SNRdBTest, MDR_Stage1_SNR, color='r', linestyle='-', marker="*", markersize=5)
    plt.semilogy(SNRdBTest, MDR_Stage2_SNR, color='b', linestyle='-', marker="o", markersize=5)
    plt.legend(["Stage1", "Stage 2"])
    plt.xlabel("SNR (dB)")
    plt.ylabel("Miss Detection Rate (MDR)")
    plt.grid()
    tracer.store(plt.gcf(), "MDR.png")

if RUN_FAR == True:
    pass


