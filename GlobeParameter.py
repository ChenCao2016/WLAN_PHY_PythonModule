L_preambleBandth = 20e6;
LSTF_length = 8e-6;
LSTF_windowRepeat = 10.0;
LLTF_length = 8e-6;
LLTF_cp = 1.6e-6;
LLTF =[0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0];
L_cp = 0.8e-6;
LSIG_length = 4e-6;

class pilot64:
    index = [11,25,39,53];
    symbol = [1,1,1,-1];
    num = 4;

def conj(point):
    return point.real - 1j*point.imag;