import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
error_params = {
    'ota_gain': np.inf,      
    'ota_offset':0.0,     
    'cap_mismatch':[0.0,0.0,0.0,0.0], 
    'comp_offset':0.0,      
    'oa_nonlinear_gain':0.0,
    'oa_bw': np.inf          
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
def residue_amplifier(vin,voltage_dac,c_total,C_f,err=error_params):
    G_ideal=c_total/C_f
    if err['ota_gain']< np.inf:
        G=G_ideal*(err['ota_gain']/(err['ota_gain']+1))
    else:
        G=G_ideal
    resid=G *(vin - voltage_dac) + err['ota_offset']
    resid = resid *(1+ err['oa_nonlinear_gain']*resid**2)
    return resid
def mdac_2_5_stages(vin,vref=1.0,err=error_params):
    x=sample_and_hold(vin)
    code = comparator(x,vref,err)
    voltage_dac, c_total, C_f = capacitive_banks(code,vref,err)
    resid = residue_amplifier(x,voltage_dac,c_total,C_f,err)
    return resid,code
def digital_packer(codes,radix=4):
    raw = 0
    for c in codes:
        raw= raw*radix +(c+2)
    return raw
def pipeline_adc(vin, stages=6, vref=1.0, err=error_params):
    x=vin
    codes =[]
    for _ in range(stages):
        x,c = mdac_2_5_stages(x, vref, err)
        codes.append(c)
    raw =digital_packer(codes, radix=4)
    return raw
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

fs = 500e6; N = 4096; C = 1638
f0 = C/N*fs
t = np.arange(N)/fs
vin = np.sin(2*np.pi*f0*t)

raw = np.array([pipeline_adc(v,6,1.0,error_params) for v in vin])
maxraw= 4**6 - 1
vout = raw/maxraw* 2 - 1
SNDR = calc_snrd_fft(vout, fs)
print("SNDR with zero errors in dB is ", SNDR)

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
