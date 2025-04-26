import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
error_params = {
    'ota_gain': 150.0,                    
    'ota_offset': 0.10,                       
    'cap_mismatch':[0.015, -0.012, 0.018, -0.010], 
    'comp_offset':0.09,                      
    'oa_nonlinear_gain':0.008,                    
    'oa_bw':70e6,                        
    'fs':500e6
}
def sample_and_hold(vin):
    return vin
def comparator(vin,vref=1.0,err=error_params):
    vin_cmp=vin+err['comp_offset']
    code =int(np.clip(np.round(vin_cmp/(vref/2)), -2, 2))
    return code
def capacitive_banks(code,vref=1.0,err=error_params):
    cm=err['cap_mismatch']
    C1=2*(1+cm[0]);C2=1*(1+cm[1]);C3=1*(1+cm[2]);C4=1*(1+cm[3])
    c_total=C1+C2 +C3
    voltage_dac=(code/2)*vref
    return voltage_dac, c_total,C4
def digital_packer(codes,radix=4):
    raw = 0
    for c in codes:
        raw= raw*radix +(c+2)
    return raw
def pipeline_adc(vin_array, err, stages=6, vref=1.0):
    fs=err['fs']
    alpha= None
    if err['oa_bw'] <np.inf:
        alpha = np.exp(-2*np.pi*err['oa_bw']/fs)
    N = len(vin_array)
    raw_codes=np.zeros(N, dtype=int)
    prev_resid = np.zeros(stages)
    for i in range(N):
        x = vin_array[i]
        codes = []
        for j in range(stages):
            vsh = x
            code = comparator(vsh, vref, err)
            codes.append(code)
            vdac, c_total, c_f = capacitive_banks(code, vref, err)
            G =c_total/c_f
            resid_ideal=G *(vsh-vdac)+err['ota_offset']
            resid_ideal*=(1+err['oa_nonlinear_gain']*resid_ideal**2)
            if alpha is not None:
                resid=alpha*prev_resid[j]+(1-alpha)*resid_ideal
            else:
                resid =resid_ideal
            prev_resid[j]=resid
            x =resid
        raw_codes[i]=digital_packer(codes, radix=4)
    return raw_codes
def calc_snrd_fft(vout, fs, guard=5):
    x = vout - np.mean(vout)
    N =len(x)
    Xc =fft.fft(x, n=N)/ N
    mag2= np.abs(Xc[:N//2])**2
    mag2[0] = 0
    k0 =np.argmax(mag2)
    low,high = max(0, k0-guard), min(len(mag2)-1, k0+guard)
    Psig = np.sum(mag2[low:high+1])
    Pnoise =np.sum(mag2)-Psig
    return 10*np.log10(Psig/Pnoise)

if __name__ == "__main__":
    fs = error_params['fs']
    N = 4096
    C = 1638
    f0 = C/N * fs
    t = np.arange(N) / fs
    vin = np.sin(2 * np.pi * f0 * t)
    raw = pipeline_adc(vin, error_params, stages=6, vref=1.0)
    maxraw = 4**6 - 1
    vout = raw / maxraw * 2 - 1
    SNDR = calc_snrd_fft(vout, fs)
    print("SNDR with errors in dB is ", SNDR)
    plt.figure(figsize=(8,2))
    plt.stem(t[:len(t)//220], vin[:len(t)//220])
    plt.title("Vin")
    plt.xlabel("Time")
    plt.ylabel("Amp")
    plt.figure(figsize=(8,2))
    plt.stem(t[:len(t)//220], vout[:len(t)//220])
    plt.title("ADC Out")
    plt.xlabel("Time")
    plt.ylabel("Amp")

    X = np.abs(fft.fft((vout - np.mean(vout))*(np.hanning(N)), n=65536)[:32768])
    X[0] = 0
    freqs = np.arange(len(X)) * (fs/65536)/1e6
    plt.figure(figsize=(8,3))
    plt.plot(freqs, 20*np.log10(X))
    plt.title("PSD")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.xlim(0, fs/2/1e6)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
