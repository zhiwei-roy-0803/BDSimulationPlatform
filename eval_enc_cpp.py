from utils.CodeConstruction import PolarCodeConstructor
from PolarBDEnc.Encoder.PolarEnc import PolarEnc
from PolarBDEnc.Encoder.CRCEnc import CRCEnc
from utils.CRC import CRCEncoder
from utils.encoder import PolarEncoder
import os
import numpy as np
import time


N = 8
K = 4
codeConstructor = PolarCodeConstructor(QPath=os.path.join(os.getcwd(), "reliable_sequence.txt"))
frozenbits, messagebits, frozenbits_indicator, messagebits_indicator = codeConstructor.PW(N, K)
enc = PolarEnc(N, K, frozenbits, messagebits)
enc_py = PolarEncoder(N, K, frozenbits, messagebits)
msg = np.random.randint(low=0, high=2, size=K)

t1 = time.time()
cword = enc.encode(msg)
t2 = time.time()
print("enc time cpp = {:.6f} s".format(t2 - t1))
t1 = time.time()
cword_py = enc_py.non_system_encoding(msg)
t2 = time.time()
print("enc time py = {:.6f} s".format(t2 - t1))
print(np.all(cword == cword_py))
print(cword)

crc = CRCEnc(crc_n=24, crc_polynominal=[24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0])
crc_py = CRCEncoder(crc_n=24, crc_p=[24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0])

t1 = time.time()
cword_crc = crc.encode(msg)
t2 = time.time()
print("crc time cpp = {:.6f} s".format(t2 - t1))
t1 = time.time()
cword_py_crc, _ = crc_py.encode(msg)
t2 = time.time()
print("crc time py = {:.6f} s".format(t2 - t1))
print(np.all(cword_crc == cword_py_crc))
print(cword_crc)
print(cword_py_crc)



