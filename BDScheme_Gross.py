import numpy as np
from genenerate_PDCCH_candidate import CandidateGenerator
from PolarDecoder.Decoder.SCDecoder import SCDecoder
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
parser.add_argument("--C2", type=int, default=4)
args = parser.parse_args()
# Simulation Parameters
C2 = args.C2 # No. candidate after stage 1 SC decoding
L = args.L
# simulation parameter configuration
numSimulation = 10**5
SNRdBTest = [0, 1, 2, 3, 4]
# initialize Trace instance to maintain simulation hyper-parameters and results and images
experiment_name = "L1={:d} L2={:d} C2={:d}".format(1, L, C2)
if os.path.isdir(os.path.join(os.getcwd(), "BD_Gross", experiment_name)):
    shutil.rmtree(os.path.join(os.getcwd(), "BD_Gross", experiment_name))
tracer = Tracer('BD_Gross').attach(experiment_name)
configure = {"L": L,
             "C2": C2,
             "numSimulation":numSimulation,
             "SNRdBTest":SNRdBTest}
tracer.store(Config(configure))
# Initialize PDCCH candidate generator and corresponding decoders
PDCCHGenerator = CandidateGenerator()
CRCEncoder = CRCEnc(PDCCHGenerator.numCRCBits, PDCCHGenerator.crcPoly)
SCDecoders = []
SCLDecoders = []
for i in range(PDCCHGenerator.numAggregationLevel):
    N = PDCCHGenerator.codewordLengthPerAggregationLevel[i]
    for m in range(2):
        K = PDCCHGenerator.numInformationBits[m] +  PDCCHGenerator.numCRCBits
        frozenbits_indicator = PDCCHGenerator.frozenbits_indicator_set[i][m]
        messagebits_indicator = PDCCHGenerator.messagebits_indicator_set[i][m]
        SCDec = SCDecoder(N=N, K=K, frozen_bits=frozenbits_indicator, message_bits=messagebits_indicator)
        SCLDec = SCLDecoder(N=N, K=K, L=L, frozen_bits=frozenbits_indicator, message_bits=messagebits_indicator)
        SCDecoders.append(SCDec)
        SCLDecoders.append(SCLDec)

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
        # SC decoding for each candidate
        passedNoisyCodeWord = []
        passedIndex = []
        passedDecIndex = []
        passedPMs = []

        notPassedNoisyCodeWord = []
        notPassedIndex = []
        notPassedDecIndex = []
        notPassedPMs = []
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
                    decoded_bits, PM = SCDecoders[dec_idx].decode(receive_symbols_llr)
                    dec_information_bits = decoded_bits[:-PDCCHGenerator.numCRCBits]
                    dec_crc = decoded_bits[-PDCCHGenerator.numCRCBits:]
                    crcCheck = CRCEncoder.encode(dec_information_bits)[-PDCCHGenerator.numCRCBits:]
                    crcCheck[-PDCCHGenerator.numRNTIBits:] = crcCheck[-PDCCHGenerator.numRNTIBits:] ^ RNTI
                    if np.all(crcCheck == dec_crc):
                        passedNoisyCodeWord.append(receive_symbols_llr)
                        passedDecIndex.append(dec_idx)
                        passedIndex.append(cnt)
                        passedPMs.append(PM)
                    else:
                        notPassedNoisyCodeWord.append(receive_symbols_llr)
                        notPassedDecIndex.append(dec_idx)
                        notPassedIndex.append(cnt)
                        notPassedPMs.append(PM)
                    cnt += 1
        # check whether the candidate set contain the valid candidate
        if RNTIIndex not in passedIndex:
            numStage1MissDetection += 1

        # find C2 candidates that pass the CRC check
        numPass = len(passedPMs)
        if numPass > C2:
            passedPMsPMs = np.array(passedPMs)
            argIdxPassedPMs = np.argsort(passedPMsPMs)[::-1]
            noisyCodeWordStage2 = []
            idxStage2 = []
            idxDecStage2 = []
            for idx in argIdxPassedPMs[:C2]:
                noisyCodeWordStage2.append(passedNoisyCodeWord[idx])
                idxStage2.append(passedIndex[idx])
                idxDecStage2.append(passedDecIndex[idx])
        elif numPass == C2:
            noisyCodeWordStage2 = passedNoisyCodeWord
            idxStage2 = passedIndex
            idxDecStage2 = passedDecIndex
        else:
            noisyCodeWordStage2 = passedNoisyCodeWord
            idxStage2 = passedIndex
            idxDecStage2 = passedDecIndex
            numLeft = C2 - numPass
            notPassedPMs = np.array(notPassedPMs)
            argIdxNotPassedPMs = np.argsort(notPassedPMs)
            for idx in argIdxNotPassedPMs[:numLeft]:
                noisyCodeWordStage2.append(notPassedNoisyCodeWord[idx])
                idxStage2.append(notPassedIndex[idx])
                idxDecStage2.append(notPassedDecIndex[idx])

        # -------Second stage SCL decoding--------- #
        passedIndex = []
        passedPMs = []
        for i in range(C2):
            dec_idx = idxDecStage2[i]
            decoded_bits, PM = SCLDecoders[dec_idx].decode(noisyCodeWordStage2[i])
            dec_information_bits = decoded_bits[:-PDCCHGenerator.numCRCBits]
            dec_crc = decoded_bits[-PDCCHGenerator.numCRCBits:]
            crcCheck = CRCEncoder.encode(dec_information_bits)[-PDCCHGenerator.numCRCBits:]
            crcCheck[-PDCCHGenerator.numRNTIBits:] = crcCheck[-PDCCHGenerator.numRNTIBits:] ^ RNTI
            if np.all(crcCheck == dec_crc):
                passedIndex.append(idxStage2[i])
                passedPMs.append(PM)
        numPass = len(passedIndex)
        if numPass == 0:
            numMissDetection += 1
        else:
            maxPMIndexStage2 = np.argmin(passedPMs)
            finalCandidateIndex = passedIndex[maxPMIndexStage2]
            if finalCandidateIndex != RNTIIndex:
                numMissDetection += 1
        pbar.set_description("Miss Det Stage 1 = {:d}, Miss Det Stage 2 = {:d}".format(numStage1MissDetection, numMissDetection))
        numRun += 1
        if numMissDetection >= 1000:
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
plt.legend(["Stage1", "Stage2"])
plt.xlabel("SNR (dB)")
plt.ylabel("Miss Detection Rate (MDR)")
plt.grid()
tracer.store(plt.gcf(), "MDR.png")

# Start Simulation FAR
print("Simulation for FAR")
numSimulation = 10**7
FAR_SNR = []
for SNRdB in SNRdBTest:
    SNR = 10**(SNRdB/10)  # linear scale snr
    sigma = np.sqrt(1/SNR)  # Gaussian noise variance for current EbN0
    pbar = tqdm(range(numSimulation))
    numFalseAlarm = 0
    numRun = 0
    # Start Simulation for FAR in Current Eb/N0
    for _ in pbar:
        information_bits, codewords, RNTI, RNTIIndex = PDCCHGenerator.generate_candidates(isRNTI=False)
        # -------First stage low complexity SC decoding--------- #
        # SC decoding for each candidate
        passedNoisyCodeWord = []
        passedIndex = []
        passedDecIndex = []
        passedPMs = []

        notPassedNoisyCodeWord = []
        notPassedIndex = []
        notPassedDecIndex = []
        notPassedPMs = []
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
                    receive_symbols_llr = receive_symbols * (2 / sigma ** 2)
                    decoded_bits, PM = SCDecoders[dec_idx].decode(receive_symbols_llr)
                    dec_information_bits = decoded_bits[:-PDCCHGenerator.numCRCBits]
                    dec_crc = decoded_bits[-PDCCHGenerator.numCRCBits:]
                    crcCheck = CRCEncoder.encode(dec_information_bits)[-PDCCHGenerator.numCRCBits:]
                    crcCheck[-PDCCHGenerator.numRNTIBits:] = crcCheck[-PDCCHGenerator.numRNTIBits:] ^ RNTI
                    if np.all(crcCheck == dec_crc):
                        passedNoisyCodeWord.append(receive_symbols_llr)
                        passedDecIndex.append(dec_idx)
                        passedIndex.append(cnt)
                        passedPMs.append(PM)
                    else:
                        notPassedNoisyCodeWord.append(receive_symbols_llr)
                        notPassedDecIndex.append(dec_idx)
                        notPassedIndex.append(cnt)
                        notPassedPMs.append(PM)
                    cnt += 1

        # find C2 candidates that pass the CRC check
        numPass = len(passedPMs)
        if numPass > C2:
            passedPMsPMs = np.array(passedPMs)
            argIdxPassedPMs = np.argsort(passedPMsPMs)[::-1]
            noisyCodeWordStage2 = []
            idxStage2 = []
            idxDecStage2 = []
            for idx in argIdxPassedPMs[:C2]:
                noisyCodeWordStage2.append(passedNoisyCodeWord[idx])
                idxStage2.append(passedIndex[idx])
                idxDecStage2.append(passedDecIndex[idx])
        elif numPass == C2:
            noisyCodeWordStage2 = passedNoisyCodeWord
            idxStage2 = passedIndex
            idxDecStage2 = passedDecIndex
        else:
            noisyCodeWordStage2 = passedNoisyCodeWord
            idxStage2 = passedIndex
            idxDecStage2 = passedDecIndex
            numLeft = C2 - numPass
            notPassedPMs = np.array(notPassedPMs)
            argIdxNotPassedPMs = np.argsort(notPassedPMs)
            for idx in argIdxNotPassedPMs[:numLeft]:
                noisyCodeWordStage2.append(notPassedNoisyCodeWord[idx])
                idxStage2.append(notPassedIndex[idx])
                idxDecStage2.append(notPassedDecIndex[idx])

        # -------Second stage SCL decoding--------- #
        passedIndex = []
        passedPMs = []
        for i in range(C2):
            dec_idx = idxDecStage2[i]
            decoded_bits, PM = SCLDecoders[dec_idx].decode(noisyCodeWordStage2[i])
            dec_information_bits = decoded_bits[:-PDCCHGenerator.numCRCBits]
            dec_crc = decoded_bits[-PDCCHGenerator.numCRCBits:]
            crcCheck = CRCEncoder.encode(dec_information_bits)[-PDCCHGenerator.numCRCBits:]
            crcCheck[-PDCCHGenerator.numRNTIBits:] = crcCheck[-PDCCHGenerator.numRNTIBits:] ^ RNTI
            if np.all(crcCheck == dec_crc):
                passedIndex.append(idxStage2[i])
                passedPMs.append(PM)
        numPass = len(passedIndex)
        # We do not musk CRC with RNTI, therefore if more than one candidate passes all CRC check, it's wroing
        if numPass > 0:
            numFalseAlarm += 1
        pbar.set_description("False Alarm = {:d}".format(numFalseAlarm))
        numRun += 1
        if numFalseAlarm >= 1000:
            break

    # Summary Statistic: MDR, FAR
    FAR = numFalseAlarm/numRun
    FAR_SNR.append(FAR)
    print("SNR = {:.1f} dB, FAR Stage 2 = {:.6f}".format(SNRdB, FAR))
    tracer.log("{:.6f}".format(FAR), file="FAR")

# Plot result for FAR
plt.figure(dpi=300)
plt.semilogy(SNRdBTest, FAR_SNR, color='r', linestyle='-', marker="*", markersize=5)
plt.legend(["FAR"])
plt.grid()
plt.xlabel("SNR (dB)")
plt.ylabel("False Alarm Rate (FAR)")
tracer.store(plt.gcf(), "FAR.png")


