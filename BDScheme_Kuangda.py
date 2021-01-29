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
parser.add_argument("--Omega", type=int, default=5)
parser.add_argument("--RUN_FAR", type=int, default=0)
args = parser.parse_args()

# Simulation Parameters
Omega = args.Omega # No. candidate after stage 1 SC decoding
L = args.L
RUN_FAR = args.RUN_FAR
RUN_MDR = 1 - RUN_FAR
# simulation parameter configuration
numSimulation = 10**4
SNRdBTest = [-3, -2, -1, 0, 1, 2]

# initialize Trace instance to maintain simulation hyper-parameters and results and images
if RUN_MDR == True:
    experiment_name = "MDR-L1={:d}-L2={:d}-Omega={:d}".format(2, L, Omega)
else:
    experiment_name = "FAR-L1={:d}-L2={:d}-Omega={:d}".format(2, L, Omega)
if os.path.isdir(os.path.join(os.getcwd(), "BD_Kuangda", experiment_name)):
    shutil.rmtree(os.path.join(os.getcwd(), "BD_Kuangda", experiment_name))
tracer = Tracer('BD_Kuangda').attach(experiment_name)
configure = {"L": L,
             "Omega": Omega,
             "numSimulation" : numSimulation,
             "SNRdBTest" : SNRdBTest}
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
        SCLOneDec = SCLDecoder(N=N, K=K, L=2, frozen_bits=frozenbits_indicator, message_bits=messagebits_indicator)
        CASCLDec = CASCLDecoder(N=N, K=K, A=A, L=L, frozen_bits=frozenbits_indicator, message_bits=messagebits_indicator,
                                crc_n=PDCCHGenerator.numCRCBits, crc_p=PDCCHGenerator.crcPoly)
        SCLOneDecoders.append(SCLOneDec)
        CASCLDecoders.append(CASCLDec)

if RUN_MDR == True:
    # Start Simulation MDR
    print("Simulation for MDR")
    MDR_SNR = []
    for SNRdB in SNRdBTest:
        SNR = 10**(SNRdB/10)  # linear scale snr
        sigma = np.sqrt(1/SNR)  # Gaussian noise variance for current EbN0
        pbar = tqdm(range(numSimulation))
        numMissDetection = 0
        numRun = 0
        # Start Simulation for MDR in Current Eb/N0
        for _ in pbar:
            information_bits, codewords, RNTI, RNTIIndex = PDCCHGenerator.generate_candidates(isRNTI=True)
            # -------First Stage Low Complexity SC Decoding--------- #
            noisyCodeword = []
            decRes = []
            DMs = []

            crcPassedNoisyObservation = []
            crcPassedDecoderIdx = []
            crcPassedIdx = []

            DCIWinnerNoisyObservation = []
            DCIWinnerDecoderIdx = []
            DCIWinnerIdx = []
            DCIWinnerDMs = []

            # brute force SC decoding for each candidate
            cword_offset = 0
            for i in range(PDCCHGenerator.numAggregationLevel):
                dec_offset = i * 2
                numCandidate = PDCCHGenerator.numCandidatePerAggregationLevel[i]
                N = PDCCHGenerator.codewordLengthPerAggregationLevel[i]
                for m in range(numCandidate):
                    metrics = np.zeros(2)
                    observations = []
                    for j in range(2):
                        A = PDCCHGenerator.numInformationBits[j]
                        K = A + PDCCHGenerator.numCRCBits
                        cword = codewords[cword_offset + m + j * numCandidate]
                        cword = cword.astype(np.int)
                        bpsksymbols = 1 - 2 * cword
                        receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, len(cword)))
                        receive_symbols_llr = receive_symbols * (2/sigma**2)
                        decoded_bits, metric = SCLOneDecoders[dec_offset + j].decode(receive_symbols_llr)

                        dec_information_bits = decoded_bits[:-PDCCHGenerator.numCRCBits]
                        dec_crc = decoded_bits[-PDCCHGenerator.numCRCBits:]
                        crcCheck = CRCEncoder.encode(dec_information_bits)[-PDCCHGenerator.numCRCBits:]
                        crcCheck[-PDCCHGenerator.numRNTIBits:] = crcCheck[-PDCCHGenerator.numRNTIBits:] ^ RNTI

                        metrics[j] = metric / (N - K)
                        observations.append(receive_symbols_llr)

                        if np.all(crcCheck == dec_crc):
                            crcPassedNoisyObservation.append(receive_symbols_llr)
                            crcPassedDecoderIdx.append(dec_offset + j)
                            crcPassedIdx.append(cword_offset + m + j * numCandidate)

                    # For each candidate, select the DCI format with smaller DM, this step reduce the number candidate by a half
                    argminDM = int(np.argmin(metrics))
                    DCIWinnerNoisyObservation.append(observations[argminDM])
                    DCIWinnerIdx.append(cword_offset + m + argminDM * numCandidate)
                    DCIWinnerDecoderIdx.append(dec_offset + argminDM)
                    DCIWinnerDMs.append(metrics[argminDM])

                # update codeword offset
                cword_offset += PDCCHGenerator.numCandidatePerAggregationLevel[i] * 2

            # -------Second Stage CA-SCL Decoding--------- #
            '''
            (1) If only one CRC candidate pass the CRC check: return it as the final decision.
            (2) If no candidate passes CRC check, select |Omega| candidate from the DCI surviving candidates and pass the
                candidates to the stage 2 decoder.
            (3) If more than one candidates passes the CRC check, pass all of them to the stage 2 decoder.
            '''
            if len(crcPassedIdx) == 0:
                selectedCandidateIdx = np.argsort(DCIWinnerDMs)[::-1][:Omega]
                passedIndex = []
                passedPMs = []
                for idx in selectedCandidateIdx:
                    observation = DCIWinnerNoisyObservation[idx]
                    decoderIdx = DCIWinnerDecoderIdx[idx]
                    decoded_bits, PM, isPass = CASCLDecoders[decoderIdx].decode(observation, RNTI)
                    if isPass:
                        passedIndex.append(DCIWinnerIdx[idx])
                        passedPMs.append(PM)
                numPass = len(passedIndex)
                if numPass == 0:
                    numMissDetection += 1
                else:
                    minPMIndexStage2 = int(np.argmin(passedPMs))
                    finalCandidateIndex = passedIndex[minPMIndexStage2]
                    if finalCandidateIndex != RNTIIndex:
                        numMissDetection += 1

            elif len(crcPassedIdx) == 1:
                if crcPassedIdx[0] != RNTIIndex:
                    numMissDetection += 1

            else:
                passedIndex = []
                passedPMs = []
                for idx in range(len(crcPassedIdx)):
                    observation = crcPassedNoisyObservation[idx]
                    decoderIdx = crcPassedDecoderIdx[idx]
                    decoded_bits, PM, isPass = CASCLDecoders[decoderIdx].decode(observation, RNTI)
                    if isPass:
                        passedIndex.append(crcPassedIdx[idx])
                        passedPMs.append(PM)
                numPass = len(passedIndex)
                if numPass == 0:
                    numMissDetection += 1
                else:
                    minPMIndexStage2 = int(np.argmin(passedPMs))
                    finalCandidateIndex = passedIndex[minPMIndexStage2]
                    if finalCandidateIndex != RNTIIndex:
                        numMissDetection += 1

            # Finish one simulation
            numRun += 1
            pbar.set_description("Miss Det = {:d}".format(numMissDetection))
            if numMissDetection >= 100:
                break

        # Summary Statistic for a Given SNR
        MDR = numMissDetection / numRun
        MDR_SNR.append(MDR)
        print("SNR = {:.1f} dB, MDR = {:.5f}".format(SNRdB, MDR))
        tracer.log("{:.6f}".format(MDR), file="MDR_Stage")


    # Plot result for MDR of two stages
    plt.figure(dpi=300)
    plt.semilogy(SNRdBTest, MDR_SNR, color='r', linestyle='-', marker="o", markersize=5)
    plt.legend(["MDR"])
    plt.xlabel("SNR (dB)")
    plt.ylabel("Miss Detection Rate (MDR)")
    plt.grid()
    tracer.store(plt.gcf(), "MDR.png")