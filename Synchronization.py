import cmath
import copy
import math
from DFT import *


L_preambleBandth = 20e6;
LSTF_length = 8e-6;
LSTF_windowRepeat = 10.0;
LLTF_length = 8e-6;
LLTF_cp = 1.6e-6;
LLTF =[0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0];




def conj(point):
    return point.real - 1j*point.imag;


#----------------------------------------------------------------------
# search all L-STFs in samples stream
#----------------------------------------------------------------------
#INPUT
#samples: complex number array
#samplingRate: number (Hz)
#----------------------------------------------------------------------
#OUTPUT
#LSTF_endIndex: sample index array for LSTF end index
#----------------------------------------------------------------------
def LSTF_sync(samples, samplingRate, LSTF_endIndex,sync_threshold = 0.5):
 
    try:
        LSTF_endIndex.clear();

        windowSize = int(LSTF_length/LSTF_windowRepeat*samplingRate);
        correlation = [];  #length len(samples)-2*len(windowSize)
        for i in range(windowSize):
            correlation.append(0.0);
        for i in range(windowSize, len(samples) - windowSize):
            m = 0.0;
            s = 0.0;
            for j in range(windowSize):
                q = samples[i+j]*conj(samples[i+j-windowSize]);
                m = m + abs(q);
                s = s + q;     
            a = abs(s)/m;
            correlation.append(a);
        for i in range(len(samples) - windowSize,len(samples)):
            correlation.append(0.0);

        diffCorrelation = []; #length len(samples)-3*windowSize

        for i in range(2*windowSize):
            diffCorrelation.append(0.0);
            
        for i in range(2*windowSize,len(samples)-windowSize):
            tm = 0;
            for j in range(windowSize):
                tm = tm + correlation[i+j-2*windowSize] - correlation[i+j];
            diffCorrelation.append(tm/windowSize);
        for i in range(len(samples) - windowSize,len(samples)):
            diffCorrelation.append(0.0);      

        for i in range(2*windowSize,len(diffCorrelation)-2*windowSize):
            if (diffCorrelation[i] == max(diffCorrelation[(i-2*windowSize):(i+2*windowSize)]) and diffCorrelation[i] > sync_threshold): #peak
                diffCorrelation[i] = 1;
            elif (diffCorrelation[i] == min(diffCorrelation[(i-2*windowSize):(i+2*windowSize)]) and diffCorrelation[i] < -sync_threshold): #valley
                diffCorrelation[i] = -1;
            else:
                diffCorrelation[i] = 0;

        valleyIndex = - windowSize*9;
        valleyRangeL = valleyIndex - windowSize;
        valleyRangeR = valleyIndex + windowSize + 1;

        for i in range(len(diffCorrelation)):
            if (diffCorrelation[i] == 1):
                if (-1 in diffCorrelation[(i + valleyRangeL): (i + valleyRangeR)]):
                    LSTF_endIndex.append(i);
    except Exception as err:
        print(str(err));

    return


#----------------------------------------------------------------------
# estimate frequency offset on L-STF
#----------------------------------------------------------------------
#INPUT#
#samples: complex number array
#samplingRate: number (Hz)
#LSTF_endIndex: sample index for LSTF end
#----------------------------------------------------------------------
#OUTPUT#
#LSTF_frequencyOffset: frequency offset estimated on LSTF
#----------------------------------------------------------------------
def LSTF_freqOffset(samples, samplingRate, LSTF_endIndex):
        
    try:
        endIndex = LSTF_endIndex;
        windowSize = int(LSTF_length/LSTF_windowRepeat*samplingRate);
        
        phaseArray=[];
        for j in range(windowSize):            
            for i in range(8): # the first window is not used to estimate frequency
                phaseArray.append(cmath.phase(samples[endIndex-j-i*windowSize]*conj(samples[endIndex-j-i*windowSize-windowSize])));
        
        phaseShift = sum(phaseArray)/len(phaseArray);
        LSTF_frequencyOffset = phaseShift*samplingRate/cmath.pi/2/windowSize;

    except Exception as err:
        print(str(err));
        
    return LSTF_frequencyOffset;




#----------------------------------------------------------------------
# estimate frequency offset on L-LTF
#----------------------------------------------------------------------
#INPUT#
#samples: complex number array
#samplingRate: number (Hz)
#LSTF_endIndex: sample index for LSTF end
#----------------------------------------------------------------------
#OUTPUT#
#LLTF_frequencyOffset: frequency offset estimated on LLTF
#----------------------------------------------------------------------
def LLTF_freqOffset(samples, samplingRate, LSTF_endIndex):

    try:
        endIndex = LSTF_endIndex + int(samplingRate*LLTF_length);
        windowSize = int((LLTF_length-LLTF_cp)/2*samplingRate);

        phaseArray=[];

        for j in range(windowSize):            
            phaseArray.append(cmath.phase(samples[endIndex-j]*conj(samples[endIndex-j-windowSize])));

        phaseShift = sum(phaseArray)/len(phaseArray);
        LLTF_frequencyOffset = phaseShift*samplingRate/cmath.pi/2/windowSize;
        
    except Exception as err:
        print(str(err));

    return LLTF_frequencyOffset;


#----------------------------------------------------------------------
# compensate frequency offset on the input samples
#----------------------------------------------------------------------
#INPUT#
#samples: original sample, complex number array
#samplingRate: number (Hz)
#frequency: frequency offset to compensate (Hz)
#newSamples: compensated samples, complex number arry
#----------------------------------------------------------------------
def freqCompensate(samples,sampleRate,frequency,newSamples):   
    
    try:
        newSamples.clear();
        length = len(samples);
        for i in range(length):
            newSamples.append(samples[i] * cmath.exp(-1j*2*cmath.pi*frequency*i/sampleRate));
                
    except Exception as err:
        print(str(err));

    return;


#----------------------------------------------------------------------
# estimate frequency offset on L-LTF
#----------------------------------------------------------------------
#INPUT#
#samples: complex number array
#samplingRate: number (Hz)
#LSTF_endIndex: sample index for LSTF end
#LLTF_channel: estimated based on L-LTF
#----------------------------------------------------------------------
def LLTF_channelEstimate(samples,samplingRate,LSTF_endIndex,LLTF_channel):
    try:
        LLTF_channel.clear();

        startIndex = LSTF_endIndex + int(samplingRate*LLTF_cp) + 1;
        windowSize = int((LLTF_length-LLTF_cp)/2*samplingRate);
        ratio = int(samplingRate/L_preambleBandth);

        temp1 = [];
        DFT(samples[startIndex:startIndex + windowSize: ratio], temp1);
        temp2 = [];
        DFT(samples[startIndex + windowSize:startIndex + 2*windowSize: ratio], temp2);
            

        for i in range(len(LLTF)):
            if LLTF[i] == 0:
                LLTF_channel.append(0.0);
            else:
                LLTF_channel.append((temp1[i]+temp2[i])/2.0/LLTF[i]);


    except Exception as err:
        print(str(err));

    return;
