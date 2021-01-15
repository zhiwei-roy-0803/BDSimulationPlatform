import numpy as np
import os
from PolarBDEnc.Encoder.PolarEnc import PolarEnc
from PolarBDEnc.Encoder.CRCEnc import CRCEnc
from utils.CodeConstruction import PolarCodeConstructor


class CandidateGenerator():

    def __init__(self):
        self.PWPath = os.path.join(os.getcwd(), "reliable_sequence.txt")
        self.numAggregationLevel = 3
        self.numCandidates = 44
        self.numRNTIBits = 16
        self.numCandidatePerAggregationLevel = np.array([6, 6, 10])
        self.codewordLengthPerAggregationLevel = np.array([128, 256, 512])
        self.maxInformationLength = 140
        self.numInformationBits = np.array([32, 57])
        self.numCRCBits = 24
        self.crcPoly = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]
        self.polarEncoders = []
        self.codeConstructor = PolarCodeConstructor(QPath=self.PWPath)
        self.crcEncoder = CRCEnc(self.numCRCBits, self.crcPoly)
        self.initialize_encoders()


    def initialize_encoders(self):
        self.frozenbits_set = []
        self.messagebits_set = []
        self.frozenbits_indicator_set = []
        self.messagebits_indicator_set = []
        for i in range(self.numAggregationLevel):
            N = self.codewordLengthPerAggregationLevel[i]
            encoder = []
            frozenbits_set = []
            messagebits_set = []
            frozenbits_indicator_set = []
            messagebits_indicator_set = []
            for A in self.numInformationBits:
                K = A + self.numCRCBits
                frozenbits, messagebits, frozenbits_indicator, messagebits_indicator = self.codeConstructor.PW(N, K)
                frozenbits_set.append(frozenbits)
                messagebits_set.append(messagebits)
                frozenbits_indicator_set.append(frozenbits_indicator)
                messagebits_indicator_set.append(messagebits_indicator)
                encoder.append(PolarEnc(N, K, frozenbits, messagebits))
            self.polarEncoders.append(encoder)
            self.frozenbits_set.append(frozenbits_set)
            self.messagebits_set.append(messagebits_set)
            self.frozenbits_indicator_set.append(frozenbits_indicator_set)
            self.messagebits_indicator_set.append(messagebits_indicator_set)


    def generate_candidates(self, isRNTI=True):
        codewords = []
        information_bits = []
        if isRNTI:
            RNTIIndex = np.random.randint(low=0, high=self.numCandidates)
        else:
            RNTIIndex = -1
        RNTI = np.random.randint(low=0, high=2, size=self.numRNTIBits, dtype=np.uint8)
        cnt = 0
        for i in range(self.numAggregationLevel):
            for m in range(2):
                A = self.numInformationBits[m]
                for j in range(self.numCandidatePerAggregationLevel[i]):
                    msg = np.random.randint(low=0, high=2, size=A)
                    msg_crc = self.crcEncoder.encode(msg)
                    if cnt == RNTIIndex:
                        msg_crc[-self.numRNTIBits:] ^= RNTI
                    codeword = self.polarEncoders[i][m].encode(msg_crc)
                    codewords.append(codeword)
                    information_bits.append(msg)
                    cnt += 1
        return information_bits, codewords, RNTI, RNTIIndex