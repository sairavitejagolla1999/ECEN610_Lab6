import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
error_params = {
    'ota_gain':np.inf,
    'ota_offset':0.0,
    'cap_mismatch':[0.0,0.0,0.0,0.0],
    'comp_offset':0.0,
    'oa_nonlinear_gain':[0.10,0.20,0.15,0.10],
    'oa_bw':np.inf,
    'fs':500e6
}
ideal_params = error_params.copy()
ideal_params['oa_nonlinear_gain'] = [0,0,0,0]
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
def pipeline_adc_time(vin_array, err, stages, vref):
    fs=err['fs']
    alpha= None
    if err['oa_bw']<np.inf:
        alpha =np.exp(-2*np.pi*err['oa_bw']/fs)
    N =len(vin_array)
    raw =np.zeros(N, int)
    stagesC= np.zeros((N, stages))
    prev_r= np.zeros(stages)
    a2,a3,a4,a5=err['oa_nonlinear_gain']
    for i in range(N):
        x=vin_array[i]
        codes=[]
        for j in range(stages):
            code=comparator(x, vref, err)
            codes.append(code)
            vdac,tot,c_f = capacitive_banks(code, vref, err)
            r0 =(tot/c_f)*(x-vdac) + err['ota_offset']
            r =r0 +a2*r0**2+a3*r0**3+a4*r0**4 +a5*r0**5
            if alpha is not None:
                r = alpha*prev_r[j] + (1-alpha)*r
            prev_r[j] = r
            x = r
        raw[i] = digital_packer(codes, 4)
        stagesC[i,:] = codes
    return raw, stagesC
def calc_snrd_fft(vout, fs, guard=1):
    N=len(vout)
    x=vout-np.mean(vout)
    w=np.hanning(N)
    X=fft.fft(x*w)
    mag2=np.abs(X[:N//2])**2
    mag2[0]=0
    k0=np.argmax(mag2)
    idx=np.arange(k0-guard, k0+guard+1)
    idx=idx[(idx>=0)&(idx<len(mag2))]
    Psig=mag2[idx].sum()
    Pnoise=mag2.sum() -Psig
    return 10*np.log10(Psig/Pnoise)
def plot_psd(v, fs, title):
    N=len(v)
    w=np.hanning(N)
    X=fft.fft((v-np.mean(v))*w, n=65536)[:32768]
    mag=np.abs(X)
    mag_db=20*np.log10(np.maximum(mag,1e-12))
    freqs=np.arange(len(mag_db))*(fs/65536)/1e6
    plt.figure(figsize=(6,3))
    plt.plot(freqs, mag_db)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.xlim(0, fs/2/1e6)
    plt.grid(True)
    plt.tight_layout()
fs = 500e6
N = 4096
C = int(200e6/fs * N)
t = np.arange(N)/fs
signal = np.sin(2*np.pi*(C/N)*fs * t)
noise_var = 1/10**(80/10)
noise = np.sqrt(noise_var)*np.random.randn(N)
vin = signal + noise

required_bw = -fs/(2*np.pi)*np.log(1.0/2**13)
error_params['oa_bw'] = 0.8*required_bw
ideal_params['oa_bw'] = error_params['oa_bw']

raw_ideal, _ =pipeline_adc_time(vin,ideal_params,4,1.0)
raw_uncal, stage_unc =pipeline_adc_time(vin, error_params,4,1.0)
maxraw=4**4 -1

vout_uncal=raw_uncal/maxraw*2 - 1
snr_uncal=calc_snrd_fft(vout_uncal, fs)
mu_n=0.5
orders=[1,2,3,4,5]
conv_iters=[]
max_snrs=[]
snr_hists=[]
it_hists=[]
w_list=[]
for p in orders:
    Xfeat = np.hstack([stage_unc**k for k in range(1,p+1)])
    w = np.zeros(Xfeat.shape[1])
    snr_hist = []
    it_hist = []
    for n in range(N):
        y = raw_uncal[n] + w.dot(Xfeat[n])
        e = raw_ideal[n] - y
        w += mu_n * e * Xfeat[n] / (Xfeat[n].dot(Xfeat[n]) + 1e-6)
        if n%50==0:
            vcal = (raw_uncal + Xfeat.dot(w))/maxraw*2 - 1
            snr_hist.append(calc_snrd_fft(vcal,fs))
            it_hist.append(n)
    w_list.append(w)
    max_snr = max(snr_hist)
    iter_snr = it_hist[snr_hist.index(max_snr)]
    conv_iters.append(iter_snr)
    max_snrs.append(max_snr)
    snr_hists.append(snr_hist)
    it_hists.append(it_hist)

print("SNR uncalibrated",snr_uncal)
for p,mi,ms in zip(orders,conv_iters,max_snrs):
    print(f"Order {p}: Max SNR = {ms:.2f} dB at iteration {mi}")

plot_psd(vout_uncal, fs, "PSD: Uncalibrated")
for p, w in zip(orders, w_list):
    Xfeat = np.hstack([stage_unc**k for k in range(1,p+1)])
    vcal = (raw_uncal + Xfeat.dot(w))/maxraw*2 - 1
    plot_psd(vcal, fs, f"PSD: Order {p}")
plt.show()
