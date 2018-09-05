from math import *

def Encoder(input,output,rate):
    ret = 0;

    output.clear();
    ShiftRegister = [0,0,0,0,0,0,0];
    BitStreamLen = len(input);


    #if rate == "1/2": (default)
    MaskLen = 1;
    PunctureMaskA = [1];
    PunctureMaskB = [1];

    if rate == "3/4":
        MaskLen = 8;
        PunctureMaskA = [1,1,0,1,1,0,1,1,0];
        PunctureMaskB = [1,0,1,1,0,1,1,0,1];

    if rate == "2/3":
        MaskLen = 6;
        PunctureMaskA = [1,1,1,1,1,1];
        PunctureMaskB = [1,0,1,0,1,0];

    
    MaskCounter = 0;
    for i in range(BitStreamLen):

        for j in range(6,0,-1):
            ShiftRegister[j] = ShiftRegister[j-1];
        ShiftRegister[0] = input[i];
        a = ShiftRegister[0]^ShiftRegister[2]^ShiftRegister[3]^ShiftRegister[5]^ShiftRegister[6];
        b = ShiftRegister[0]^ShiftRegister[1]^ShiftRegister[2]^ShiftRegister[3]^ShiftRegister[6];

        if PunctureMaskA[MaskCounter] == 1:
            output.append(a);
        
        if PunctureMaskB[MaskCounter] == 1:
            output.append(b);

        MaskCounter = (MaskCounter+1) % MaskLen;
        
    return ret;
    

def Decoder(input,output,rate):
    ret = 0;

    ShiftRegister = [999,999,999,999,999,999,999];
    inputA = [];
    inputB = [];

    output.clear();

    #if rate == "1/2": (default)
    BitStreamLen = int(len(input)/2); 
    MaskLen = 1;
    PunctureMaskA = [1];
    PunctureMaskB = [1];
    
    if rate == "3/4":
        BitStreamLen = int(len(input)*3/4); 
        MaskLen = 8;
        PunctureMaskA = [1,1,0,1,1,0,1,1,0];
        PunctureMaskB = [1,0,1,1,0,1,1,0,1];

    if rate == "2/3":
        BitStreamLen = int(len(input)*2/3); 
        MaskLen = 6;
        PunctureMaskA = [1,1,1,1,1,1];
        PunctureMaskB = [1,0,1,0,1,0];
    

    #interpolate the puncture
    MaskCounter = 0;
    currentIndex = 0;
    for i in range(BitStreamLen):

        if PunctureMaskA[MaskCounter] == 1:
            inputA.append(input[currentIndex]);
            currentIndex = currentIndex + 1;
        else:
            inputA.append(0);  #LLR = log2(1);

        if PunctureMaskB[MaskCounter] == 1:
            inputB.append(input[currentIndex]);
            currentIndex = currentIndex + 1;
        else:
            inputB.append(0);  #LLR = log2(1);

        MaskCounter = (MaskCounter+1) % MaskLen;


    #forward propagation
    for i in range(BitStreamLen - 6): # the last 6 bits have to tail (six "0")

        for j in range(6,0,-1):
            ShiftRegister[j] = ShiftRegister[j-1];

        a = 2*atanh(tanh(ShiftRegister[2]/2)*tanh(ShiftRegister[3]/2)*tanh(ShiftRegister[5]/2)*tanh(ShiftRegister[6]/2)*tanh(inputA[i]/2));
        b = 2*atanh(tanh(ShiftRegister[1]/2)*tanh(ShiftRegister[2]/2)*tanh(ShiftRegister[3]/2)*tanh(ShiftRegister[6]/2)*tanh(inputB[i]/2));

        LLR = a + b;

        ShiftRegister[0] = LLR;

        output.append(LLR);

    for i in range(6): # the last 6 bits have to tail (six "0"s)
        output.append(999);



    #backward progagation
    ShiftRegister = [999,999,999,999,999,999,999];
   
    for i in range(BitStreamLen - 7,-1,-1): # the last 6 bits have to tail (six "0")

        for j in range(0,6,1):
            ShiftRegister[j] = ShiftRegister[j + 1];

        a = 2*atanh(tanh(ShiftRegister[0]/2)*tanh(ShiftRegister[2]/2)*tanh(ShiftRegister[3]/2)*tanh(ShiftRegister[5]/2)*tanh(inputA[i]/2));
        b = 2*atanh(tanh(ShiftRegister[0]/2)*tanh(ShiftRegister[1]/2)*tanh(ShiftRegister[2]/2)*tanh(ShiftRegister[3]/2)*tanh(inputB[i]/2));

        LLR = a + b;

        ShiftRegister[6] = LLR;

        output[i] = output[i] + LLR;


    return ret;


if __name__ == "__main__":
    input = [1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0];
    output = [];
    Encode(input,output,"2/3");
    print(output);

