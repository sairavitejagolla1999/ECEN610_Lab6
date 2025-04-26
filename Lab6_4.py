import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
error_params = {
    'ota_gain': np.inf,
    'ota_offset':0.0,
    'cap_mismatch':[0.0,0.0,0.0,0.0],
    'comp_offset':0.0,
    'oa_nonlinear_gain':0.5,
    'oa_bw':np.inf,
    'fs':500e6
}
ideal_params=error_params.copy()
ideal_params['oa_nonlinear_gain']=0.0
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
def pipeline_adc_time(vin_array, err, stages=4, vref=1.0):
    fs = err['fs']
    alpha = None
    if err['oa_bw']< np.inf:
        alpha= np.exp(-2*np.pi*err['oa_bw']/fs)
    N  = len(vin_array)
    raw = np.zeros(N, dtype=int)
    stagesC= np.zeros((N, stages))
    prev_r = np.zeros(stages)
    for i in range(N):
        x =vin_array[i]
        codes =[]
        for j in range(stages):
            code =comparator(x, vref, err)
            codes.append(code)
            vdac,ctot,cf=capacitive_banks(code,vref,err)
            G=ctot/cf
            r=G*(x-vdac)+err['ota_offset']
            r *=(1+err['oa_nonlinear_gain']*r*r)
            if alpha is not None:
                r= alpha*prev_r[j] + (1-alpha)*r
            prev_r[j]=r
            x = r
        raw[i]= digital_packer(codes, 4)
        stagesC[i,:]=codes
    return raw, stagesC
def calc_snrd_fft(vout, fs, guard=1):
    N =len(vout)
    x =vout - np.mean(vout)
    w =np.hanning(N)
    X =fft.fft(x*w)
    mag2= np.abs(X[:N//2])**2
    mag2[0] = 0
    k0 = np.argmax(mag2)
    idx= np.arange(k0-guard, k0+guard+1)
    idx = idx[(idx>=0)&(idx<len(mag2))]
    Psig =mag2[idx].sum()
    Pnoise=mag2.sum() - Psig
    return 10*np.log10(Psig/Pnoise)
fs = 500e6
N = 4096
C = 1638
f0 = C/N * fs
t = np.arange(N) / fs
vin = np.sin(2 * np.pi * f0 * t)
raw_ideal, _ = pipeline_adc_time(vin, ideal_params, 4, 1.0)
maxraw = 4**4 - 1
raw_uncalibrated, stage_unc = pipeline_adc_time(vin, error_params, 4, 1.0)
vout_uncal = raw_uncalibrated/maxraw*2 - 1
snr_uncal = calc_snrd_fft(vout_uncal, fs)
# max_norm = max(snr_norm)
# limit_db = max_norm
mu = 2e-4
w_std = np.zeros(4)
snr_std = []
it_std = []
for n in range(N):
    y= raw_uncalibrated[n] + w_std.dot(stage_unc[n])
    e= raw_ideal[n] - y
    w_std+= mu * e * stage_unc[n]
    if n % 50 == 0:
        vcal= (raw_uncalibrated + stage_unc.dot(w_std)) / maxraw * 2 - 1
        snr_std.append(calc_snrd_fft(vcal, fs))
        it_std.append(n)
mu_n=0.5
w_normalized=np.zeros(4)
snr_norm =[]
it_norm =[]

for n in range(N):
    xvec=stage_unc[n]
    y=raw_uncalibrated[n] + w_normalized.dot(xvec)
    e=raw_ideal[n]-y
    w_normalized += mu_n * e * xvec / (xvec.dot(xvec) + 1e-6)
    if n % 50 == 0:
        vcal = (raw_uncalibrated + stage_unc.dot(w_normalized)) / maxraw * 2 - 1
        snr_norm.append(calc_snrd_fft(vcal, fs))
        it_norm.append(n)
max_norm = max(snr_norm)

iter_norm = it_norm[snr_norm.index(max_norm)]
max_std = max(snr_std)
iter_std = it_std[snr_std.index(max_std)]
print("SNR uncalibrated  dB",snr_uncal)
print(f"Max SNR normalized LMS    = {max_norm:.2f} dB at iteration {iter_norm}")
print(f"Max SNR standard LMS      = {max_std:.2f} dB at iteration {iter_std}")
def plot_psd(v,title):
    N = len(v)
    w = np.hanning(N)
    X = fft.fft((v-np.mean(v))*w, n=65536)[:32768]
    mag = np.abs(X)
    mag_db = 20*np.log10(np.maximum(mag,1e-12))
    freqs = np.arange(len(mag_db))*(fs/65536)/1e6
    plt.figure(figsize=(8,3))
    plt.plot(freqs, mag_db)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.xlim(0, fs/2/1e6)
    plt.grid(True)
    plt.tight_layout()
plot_psd(vout_uncal,"PSD: Uncalibrated")
plot_psd(vcal,"PSD: Calibrated")
plt.figure()
plt.plot(it_std, snr_std, label="Std LMS")
plt.plot(it_norm, snr_norm, label="Norm LMS")
# plt.axhline(limit_db, color='k', linestyle='--', label=f"{limit_db:.1f} dB Limit")
plt.axhline(max_norm, color='k', linestyle='--', label=f"{max_norm:.2f} dB Limit")
plt.xlabel("Iteration")
plt.ylabel("SNDR (dB)")
plt.legend()
plt.grid()
plt.show()
