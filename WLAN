import cmath
import copy

class WLAN:
    conf = {
     "sampling_rate": 80e6,
     "sync_threshold": 0.5,
     "LSTF_endIndex": 0
    };

    result = {
     "LSTF_diffCorrelation": [],
     "LSTF_endIndex": [],
     "LSTF_frequencyOffset": -999,
    }; 

    LSTF_length = 8e-6;
    LSTF_windowRepeat = 10.0;

    def conj(self, point):
        return point.real - 1j*point.imag;
    


    def LSTF_sync(self, samples):
        try:
            windowSize = int(self.LSTF_length/self.LSTF_windowRepeat*self.conf["sampling_rate"]);

            correlation = [];  #length len(samples)-2*len(windowSize)
            for i in range(windowSize):
                correlation.append(0.0);
            for i in range(windowSize, len(samples) - windowSize):
                m = 0.0;
                s = 0.0;
                for j in range(windowSize):
                    q = samples[i+j]*self.conj(samples[i+j-windowSize]);
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
                if (diffCorrelation[i] == max(diffCorrelation[(i-2*windowSize):(i+2*windowSize)]) and diffCorrelation[i] > self.conf["sync_threshold"]): #peak
                    diffCorrelation[i] = 1;
                elif (diffCorrelation[i] == min(diffCorrelation[(i-2*windowSize):(i+2*windowSize)]) and diffCorrelation[i] < -self.conf["sync_threshold"]): #valley
                    diffCorrelation[i] = -1;
                else:
                    diffCorrelation[i] = 0;

            valleyIndex = - windowSize*9;
            valleyRangeL = valleyIndex - windowSize;
            valleyRangeR = valleyIndex + windowSize + 1;

            for i in range(len(diffCorrelation)):
                if (diffCorrelation[i] == 1):
                    if (-1 in diffCorrelation[(i + valleyRangeL): (i + valleyRangeR)]):
                        self.result["LSTF_endIndex"].append(i);

        except Exception as err:
            print (str(err));

        return;

    def LSTF_freqOffset(self,samples):
        
        try:

            endIndex = self.conf["LSTF_endIndex"];
            windowSize = int(self.LSTF_length/self.LSTF_windowRepeat*self.conf["sampling_rate"]);
        
            phaseArray=[];
            for j in range(windowSize):            
                for i in range(8): # the first window is not used to estimate frequency
                    phaseArray.append(cmath.phase(samples[endIndex-j-i*windowSize]*self.conj(samples[endIndex-j-i*windowSize-windowSize])));
        
            phaseShift = sum(phaseArray)/len(phaseArray);
            freqOffset = phaseShift*self.conf["sampling_rate"]/cmath.pi/2/windowSize;

            self.result["LSTF_frequencyOffset"] = freqOffset;
        except Exception as err:
            print(str(err));
        
        return



        











def unitTest():
    import array

    file = open("11AX.dat","rb");
    
    IQdata = array.array("f")
    IQdata.fromstring(file.read());
    file.close();


    Idata=[];
    Qdata=[];
    for i in range(0,len(IQdata),2):
        Idata.append(IQdata[i]);
    for i in range(1,len(IQdata),2):
        Qdata.append(IQdata[i]);

    samples=[];
    for i in range(0, len(Idata)):
        samples.append(Idata[i] + 1j*Qdata[i]);
    
    print(len(samples));


    receiver1 = WLAN();
    #receiver1.sync(samples);
    #result = receiver1.result["LSTF_diffCorrelation"];
    #print (receiver1.result["LSTF_endIndex"]);

    receiver1.conf["LSTF_endIndex"] = 8651;
    receiver1.LSTF_freqOffset(samples);

    #file = open("WLAN.csv","w+");
    #string = "\n".join(str(e) for e in result);
    #file.write(string);
    #file.close();

    return;

if __name__=="__main__":
    unitTest();
