import serial
import time
import numpy as np

# 共享队列接口函数
def radar_data_collector(radar_queue):
    configFileName = '1443config.cfg'

    CLIport = serial.Serial('COM3', 115200)
    Dataport = serial.Serial('COM4', 921600)

    def serialConfig(configFileName):
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:
            CLIport.write((i + '\n').encode())
            time.sleep(0.01)

    def parseConfigFile(configFileName):
        configParameters = {}
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:
            splitWords = i.split(" ")
            if "profileCfg" in splitWords[0]:
                startFreq = int(float(splitWords[2]))
                idleTime = int(splitWords[3])
                rampEndTime = float(splitWords[5])
                freqSlopeConst = float(splitWords[8])
                numAdcSamples = int(splitWords[10])
                numAdcSamplesRoundTo2 = 1
                while numAdcSamples > numAdcSamplesRoundTo2:
                    numAdcSamplesRoundTo2 *= 2
                digOutSampleRate = int(splitWords[11])
            elif "frameCfg" in splitWords[0]:
                chirpStartIdx = int(splitWords[1])
                chirpEndIdx = int(splitWords[2])
                numLoops = int(splitWords[3])
                framePeriodicity = int(splitWords[5])

        numTxAnt = 3
        numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
        configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
        configParameters["numRangeBins"] = numAdcSamplesRoundTo2
        configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / \
            (2 * freqSlopeConst * 1e12 * numAdcSamplesRoundTo2)
        configParameters["dopplerResolutionMps"] = 3e8 / \
            (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
        return configParameters

    def readAndParseData14xx(Dataport, configParameters):
        magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
        byteBuffer = np.zeros(2**15, dtype='uint8')
        byteBufferLength = 0
        maxBufferSize = 2**15

        OBJ_STRUCT_SIZE_BYTES = 12
        MMWDEMO_UART_MSG_DETECTED_POINTS = 1

        detObj = {}
        print("[INFO] Radar data collector started...")
        while True:
            readBuffer = Dataport.read(Dataport.in_waiting)
            byteVec = np.frombuffer(readBuffer, dtype='uint8')
            byteCount = len(byteVec)
            if (byteBufferLength + byteCount) < maxBufferSize:
                byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
                byteBufferLength += byteCount

            if byteBufferLength > 16:
                possibleLocs = np.where(byteBuffer == magicWord[0])[0]
                startIdx = [loc for loc in possibleLocs if np.all(byteBuffer[loc:loc+8] == magicWord)]
                if startIdx:
                    byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                    byteBufferLength -= startIdx[0]

                    word = [1, 2**8, 2**16, 2**24]
                    totalPacketLen = np.matmul(byteBuffer[12:12+4], word)
                    if byteBufferLength >= totalPacketLen:
                        idX = 0
                        idX += 8  # skip magic
                        idX += 4  # version
                        totalPacketLen = np.matmul(byteBuffer[idX:idX+4], word)
                        idX += 4
                        idX += 4  # platform
                        frameNumber = np.matmul(byteBuffer[idX:idX+4], word)
                        idX += 4
                        idX += 4  # timeCpuCycles
                        numDetectedObj = np.matmul(byteBuffer[idX:idX+4], word)
                        idX += 4
                        numTLVs = np.matmul(byteBuffer[idX:idX+4], word)
                        idX += 4

                        for _ in range(numTLVs):
                            tlv_type = np.matmul(byteBuffer[idX:idX+4], word)
                            idX += 4
                            tlv_length = np.matmul(byteBuffer[idX:idX+4], word)
                            idX += 4
                            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                                word16 = [1, 2**8]
                                tlv_numObj = np.matmul(byteBuffer[idX:idX+2], word16)
                                idX += 2
                                tlv_xyzQFormat = 2**np.matmul(byteBuffer[idX:idX+2], word16)
                                idX += 2

                                x = np.zeros(tlv_numObj)
                                y = np.zeros(tlv_numObj)
                                z = np.zeros(tlv_numObj)
                                rangeIdx = np.zeros(tlv_numObj)
                                dopplerIdx = np.zeros(tlv_numObj)
                                peakVal = np.zeros(tlv_numObj)

                                for i in range(tlv_numObj):
                                    rangeIdx[i] = np.matmul(byteBuffer[idX:idX+2], word16)
                                    idX += 2
                                    dopplerIdx[i] = np.matmul(byteBuffer[idX:idX+2], word16)
                                    idX += 2
                                    peakVal[i] = np.matmul(byteBuffer[idX:idX+2], word16)
                                    idX += 2
                                    x[i] = np.matmul(byteBuffer[idX:idX+2], word16)
                                    idX += 2
                                    y[i] = np.matmul(byteBuffer[idX:idX+2], word16)
                                    idX += 2
                                    z[i] = np.matmul(byteBuffer[idX:idX+2], word16)
                                    idX += 2

                                rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                                dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                                x = x / tlv_xyzQFormat
                                y = y / tlv_xyzQFormat
                                z = z / tlv_xyzQFormat

                                detObj = {
                                    "frame": int(frameNumber),
                                    "numObj": int(tlv_numObj),
                                    "range": rangeVal.tolist(),
                                    "doppler": dopplerVal.tolist(),
                                    "x": x.tolist(),
                                    "y": y.tolist(),
                                    "z": z.tolist(),
                                    "timestamp": time.time()
                                }

                                radar_queue.put(detObj)

                        shiftSize = totalPacketLen
                        byteBuffer[:byteBufferLength-shiftSize] = byteBuffer[shiftSize:byteBufferLength]
                        byteBufferLength -= shiftSize

        # time.sleep(0.1)

    try:
        serialConfig(configFileName)
        configParameters = parseConfigFile(configFileName)
        readAndParseData14xx(Dataport, configParameters)
    except KeyboardInterrupt:
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
