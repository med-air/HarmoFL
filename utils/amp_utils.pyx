cimport cython
import numpy as np
cimport numpy as np

def extract_amp(img_np):
    cdef np.ndarray[np.complex128_t, ndim=3] fft = np.fft.fft2(img_np, axes=(-2, -1))
    cdef np.ndarray[double, ndim=3] amp_np = np.abs(fft)
    return amp_np

def mutate(amp_src, amp_trg, L=0.1):
    cdef np.ndarray[double, ndim=3] a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    cdef np.ndarray[float, ndim=3] a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))
    cdef int h, w, b, c_h, c_w, h1, h2, w1, w2
    h = a_src.shape[1]
    w = a_src.shape[2]

    b = (np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def normalize(src_img, amp_trg, L=0.1):
    cdef np.ndarray[np.complex128_t, ndim=3] fft_src_np = np.fft.fft2(src_img, axes=(-2, -1) )
    cdef np.ndarray[double, ndim=3] amp_src = np.abs(fft_src_np)
    cdef np.ndarray[double, ndim=3] pha_src = np.angle(fft_src_np)
    cdef np.ndarray[double, ndim=3] amp_src_ = mutate(amp_src, amp_trg, L=L)
    cdef np.ndarray[np.complex128_t, ndim=3] fft_src_ = amp_src_ * np.exp( 1j * pha_src )
    cdef np.ndarray[np.complex128_t, ndim=3] src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    cdef np.ndarray[double, ndim=3] src_in_trg_real = np.real(src_in_trg)
    return src_in_trg_real


def process(np.ndarray[float, ndim=4] x, np.ndarray[float, ndim=3] running_amp, int momentum, bint fix_amp):
    cdef unsigned int B = x.shape[0]
    cdef unsigned int C = x.shape[1]
    cdef unsigned int W = x.shape[2]
    cdef unsigned int H = x.shape[3]

    cdef np.ndarray[float, ndim=4] amp_list = np.zeros((B,C,W,H), dtype=np.float32)
    if not fix_amp:    
        for idx in range(B):
            amp_np = extract_amp(x[idx])
            amp_list[idx,:,:,:] = amp_np
        amp_avg = np.mean(amp_list,axis=0)
        if np.sum(running_amp) == 0:
            running_amp = amp_avg
        else:
            running_amp = running_amp * (1-momentum) + amp_avg * momentum
    del amp_list
    for idx in range(B):
        x[idx] = normalize(x[idx], running_amp[:3,...], L=0)

    return x, running_amp
    