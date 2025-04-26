import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
error_params = {
    'ota_gain':np.inf,
    'ota_offset':0.0,
    'cap_mismatch':[0.015,-0.012,0.018,-0.010],
    'comp_offset':0.0,
    'oa_nonlinear_gain':[0,0,0,0],
    'oa_bw': np.inf,
    'fs': 500e6
}
ideal_params=error_params.copy()
ideal_params['cap_mismatch'] = [0,0,0,0]
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
def pipeline_adc_time(vin_array,err,stages,vref):
    fs = err['fs']
    alpha=None
    if err['oa_bw'] < np.inf:
        alpha = np.exp(-2*np.pi*err['oa_bw']/fs)
    N =len(vin_array)
    raw =np.zeros(N, int)
    stagesC=np.zeros((N, stages))
    prev_r=np.zeros(stages)
    for i in range(N):
        x =vin_array[i]
        codes=[]
        for j in range(stages):
            code =comparator(x, vref,err)
            codes.append(code)
            vdac,c_total,cf =capacitive_banks(code,vref,err)
            r =(c_total/cf)*(x -vdac) +err['ota_offset']
            if alpha is not None:
                r=alpha*prev_r[j] + (1-alpha)*r
            prev_r[j] =r
            x =r
        raw[i] = digital_packer(codes,4)
        stagesC[i,:] =codes
    return raw, stagesC
def calc_snr_multitone(vout, tone_bins):
    X = fft.fft(vout - np.mean(vout))
    mag2 = np.abs(X[:len(vout)//2])**2
    Psig = mag2[tone_bins].sum()
    Pnoise = mag2.sum() - Psig
    return 10*np.log10(Psig/Pnoise)
def plot_psd_compare(v_list, labels, fs):
    N = len(v_list[0])
    freqs = np.arange(N//2)*(fs/N)/1e6
    plt.figure(figsize=(8,4))
    for v, lab in zip(v_list, labels):
        X = fft.fft(v - np.mean(v))
        mag = np.abs(X[:N//2])
        plt.plot(freqs, 20*np.log10(np.maximum(mag,1e-12)), label=lab)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid()
fs = 500e6
N = 640
Nt = 128
t = np.arange(N)/fs
bits = np.random.choice([-1,1], Nt)
tone_bins = 2*np.arange(Nt)
signal = np.zeros(N)
for m in range(Nt):
    signal += bits[m]*np.cos(2*np.pi*tone_bins[m]/N * np.arange(N))
signal /= Nt
noise_var=np.var(signal)/10**(80/10)
noise=np.sqrt(noise_var)*np.random.randn(N)
vin =signal+noise
raw_ideal, stage_ideal = pipeline_adc_time(vin, ideal_params, 4, 1.0)
raw_uncalibrated, stage_uncalibrated = pipeline_adc_time(vin, error_params, 4, 1.0)
maxraw =4**4 - 1
vout_uncalibrated=raw_uncalibrated/maxraw*2 - 1
mu = 2e-4
w = np.zeros(4)
err_hist = np.zeros(N)
for n in range(N):
    y=raw_uncalibrated[n] + w.dot(stage_uncalibrated[n])
    e=raw_ideal[n] - y
    w += mu * e * stage_uncalibrated[n]
    err_hist[n]=e**2
raw_calibrated=raw_uncalibrated+stage_uncalibrated.dot(w)
vout_calibrated=raw_calibrated/maxraw*2 -1
snr_uncalibrated=calc_snr_multitone(vout_uncalibrated, tone_bins)
snr_calibrated=calc_snr_multitone(vout_calibrated, tone_bins)
print(f"SNR uncalibrated = {snr_uncalibrated:.2f} dB")
print(f"SNR calibrated   = {snr_calibrated:.2f} dB")
plt.figure()
plt.plot(t*1e6, vin)
plt.title("Noisy Multitone BPSK Signal")
plt.xlabel("Time(µs)")
plt.ylabel("Amplitude")
plt.grid()
plt.figure()
plt.plot(err_hist)
plt.title("LMS Calibration vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.grid()
X_uncal=fft.fft(vout_uncalibrated)
X_cal =fft.fft(vout_calibrated)
freqs=np.arange(N)*(fs/N)/1e6
plt.figure(figsize=(8,3))
plt.plot(freqs, 20*np.log10(np.maximum(np.abs(X_uncal),1e-12)))
plt.title("DFT Magnitude Uncalibrated")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.xlim(0, fs/2/1e6)
plt.grid()
plt.figure(figsize=(8,3))
plt.plot(freqs, 20*np.log10(np.maximum(np.abs(X_cal),1e-12)))
plt.title("DFT Magnitude Calibrated")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.xlim(0, fs/2/1e6)
plt.grid()

plt.show()
