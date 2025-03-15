import streamlit as st
import numpy as np
import scipy.io
import wfdb
import onnxruntime as ort
import os
import matplotlib.pyplot as plt
from scipy.signal import hilbert, spectrogram, find_peaks, butter, filtfilt
from scipy.stats import ttest_ind  # 추가된 라이브러리
import seaborn as sns

# Streamlit 설정
st.set_page_config(page_title="부정맥 감지 시스템", page_icon="❤️", layout="centered")

# ONNX 모델 로드 함수
@st.cache_resource
def load_model():
    model_path = r"C:\Users\smhrd\trainedModel.onnx"  # ONNX 모델 경로
    if not os.path.exists(model_path):
        st.error("모델 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return None
    try:
        return ort.InferenceSession(model_path)
    except Exception as e:
        st.error(f"모델 로드 중 오류: {e}")
        return None

model = load_model()





# 순시 주파수 계산 함수
def calculate_instantaneous_frequency(signal, fs=360):
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    low_cutoff = 0.5 / nyquist  
    high_cutoff = 50 / nyquist  
    b, a = butter(2, [low_cutoff, high_cutoff], btype="band")
    filtered_signal = filtfilt(b, a, signal)

    # 2.  주파수 계산
    f, t, Sxx = spectrogram(filtered_signal, fs=fs, nperseg=256, noverlap=128)
    power_center_frequency = np.sum(f[:, None] * Sxx, axis=0) / np.sum(Sxx, axis=0)

    # 3. NaN 값을 제거하고 이상치 처리
    power_center_frequency = np.nan_to_num(power_center_frequency, nan=0.0)
    power_center_frequency = np.clip(power_center_frequency, 0, fs / 2)  # Nyquist 범위 내로 제한

    return power_center_frequency

#특징 계산 함수
def calculate_features(signal, fs=360):
    # 순시 주파수 계산
    inst_freq = calculate_instantaneous_frequency(signal, fs)
    
    # 스펙트럼 엔트로피 계산
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256)
    Sxx_norm = Sxx / Sxx.sum(axis=0, keepdims=True)
    pentropy = -np.sum(Sxx_norm * np.log(Sxx_norm + 1e-10), axis=0)
    
    return inst_freq, pentropy

# 신호 및 특징 시각화 함수
def plot_features_with_heatmap(signal, inst_freq, pentropy, max_signal_samples=8000, freq_length=250, entropy_length=250):
    signal_range = min(len(signal), max_signal_samples)
    inst_freq = inst_freq[:min(freq_length, len(inst_freq))]  # 표시할 순시 주파수 샘플 수를 늘릴 수도 있음
    pentropy = pentropy[:min(entropy_length, len(pentropy))]  

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    # 원본 신호
    axs[0].plot(signal[:signal_range], label="ECG Signal")
    axs[0].set_title(f"Original ECG Signal (First {signal_range} Samples)")
    axs[0].set_xlim([0, signal_range])

    # 순시 주파수
    axs[1].plot(inst_freq, label="Instantaneous Frequency", color="green")
    axs[1].set_title(f"Instantaneous Frequency (First {freq_length} Samples)")
    axs[1].set_xlim([0, freq_length])

    # 스펙트럼 엔트로피
    axs[2].plot(pentropy, label="Spectral Entropy", color="orange")
    axs[2].set_title(f"Spectral Entropy (First {entropy_length} Samples)")
    axs[2].set_xlabel("Samples")
    axs[2].set_xlim([0, entropy_length])

    if len(pentropy) > 0:  # pentropy의 길이가 0 이상일 때만 heatmap 생성
        heatmap_data = np.reshape(pentropy, (-1, 10))
        sns.heatmap(heatmap_data, cmap="coolwarm", ax=axs[3], cbar=True)
        axs[3].set_title("Spectral Entropy Heatmap")


    st.pyplot(fig)

# 신호 예측 함수
def predict_signal(signal, model):
    if model is None:
        return "모델이 로드되지 않았습니다.", None
    

    # 신호 정규화
    mu = np.array([5.5703, 0.6202])  # 평균값 (모델 학습 시 기준값)
    sg = np.array([3.5797, 0.0789])  # 표준편차 (모델 학습 시 기준값)
    desired_length = 9000
    if len(signal) < desired_length:
        signal = np.pad(signal, (0, desired_length - len(signal)), 'constant')
    else:
        signal = signal[:desired_length]

    # 특징 생성
    inst_freq_feature = (signal - mu[0]) / sg[0]
    pentropy_feature = (signal - mu[1]) / sg[1]
    normalized_signal = np.stack((inst_freq_feature, pentropy_feature), axis=-1)
    normalized_signal = normalized_signal[np.newaxis, :, :].astype(np.float32)

    # 모델 예측
    ort_inputs = {model.get_inputs()[0].name: normalized_signal}
    try:
        ort_outs = model.run(None, ort_inputs)
        prediction_probabilities = ort_outs[0][0]
        prediction = np.argmax(prediction_probabilities)
        result = "부정맥" if prediction == 1 else "정상"
        confidence = prediction_probabilities[prediction]
        return result, confidence
    except Exception as e:
        st.error(f"예측 중 오류: {e}")
        return "오류", None

# .mat 파일 로드 함수
@st.cache_data
def load_mat_file(file):
    try:
        mat_data = scipy.io.loadmat(file)
        return mat_data['val'].flatten()  # 부동 소수점(floating-point) 데이터로 반환
    except Exception as e:
        st.error(f"MAT 파일 로드 오류: {e}")
        return None

# .hea 및 .dat 파일 로드 함수
@st.cache_data
def load_hea_dat_files(hea_file, dat_file):
    try:
        # 임시 파일 저장 경로
        temp_dir = "temp_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # .hea와 .dat 파일 저장
        hea_path = os.path.join(temp_dir, hea_file.name)
        dat_path = os.path.join(temp_dir, dat_file.name)
        with open(hea_path, "wb") as f:
            f.write(hea_file.getbuffer())
        with open(dat_path, "wb") as f:
            f.write(dat_file.getbuffer())

        # 신호 로드
        record = wfdb.rdrecord(hea_path.split(".")[0])  # 확장자 제외한 이름 전달
        return record.p_signal.flatten()  # 부동 소수점(floating-point) 데이터로 반환
    except Exception as e:
        st.error(f"HEA 및 DAT 파일 로드 오류: {e}")
        return None




def get_lowpass_filter_coefficients(cutoff=15, fs=360, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    return butter(order, normal_cutoff, btype='low', analog=False)

def lowpass_filter(signal, b=None, a=None):
    if b is None or a is None:
        b, a = get_lowpass_filter_coefficients()
    return filtfilt(b, a, signal)
    

def detect_p_wave(signal, qrs_peaks, fs=360):
    p_peaks = []
    for qrs in qrs_peaks:
        search_start = max(0, qrs - int(0.2 * fs))  # QRS 피크 기준 200ms 이전부터 탐색
        search_end = max(0, qrs - int(0.05 * fs))   # QRS 피크 기준 50ms 이전까지 탐색
        search_region = signal[search_start:search_end]

        if len(search_region) > 0:
            p_peak, _ = find_peaks(-search_region)  # 부호 반전을 통해 최소값 탐색
            if len(p_peak) > 0:
                p_peaks.append(search_start + p_peak[np.argmin(search_region[p_peak])])  # 최소값의 실제 위치 저장
    return p_peaks

def detect_qrs_peaks(signal, fs=360):
    distance = fs // 2  # 최소 간격 (초당 샘플 수)
    threshold = np.mean(signal) + 1.5 * np.std(signal)  # 동적 임계값
    peaks, _ = find_peaks(signal, distance=distance, height=threshold)
    return peaks
    axs[0, i].set_ylim(segment.min() - 0.1, segment.max() + 0.1)
    axs[1, i].set_ylim(segment.min() - 0.1, segment.max() + 0.1)


def plot_ecg_with_peaks(signal, fs=360):
    # 필터링
    filtered_signal = lowpass_filter(signal)

    # QRS 피크 검출
    qrs_peaks, _ = find_peaks(filtered_signal, distance=fs // 2, height=np.mean(filtered_signal))

    # P 파 검출
    p_peaks = detect_p_wave(filtered_signal, qrs_peaks, fs)

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(signal, label="Original ECG Signal", alpha=0.5)
    ax.plot(filtered_signal, label="Filtered ECG Signal", color="blue")
    ax.plot(qrs_peaks, filtered_signal[qrs_peaks], "x", label="QRS Peaks", color="red")
    ax.plot(p_peaks, filtered_signal[p_peaks], "o", label="P Peaks", color="green")
    ax.legend()
    ax.set_title("ECG Signal with QRS and P Peaks")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_qrs_p_wave_shapes(signal, qrs_peaks, p_peaks, fs=360):
    """
    QRS와 P 파를 확대하여 개별 파형으로 시각화합니다.
    """
    window = int(0.2 * fs)  # 각 피크 주변 200ms 범위를 확인
    fig, axs = plt.subplots(2, len(qrs_peaks[:5]), figsize=(15, 6))  # 최대 5개의 피크를 시각화
    fig.suptitle("QRS 및 P 파 확대 시각화", fontsize=16)

    for i, peak in enumerate(qrs_peaks[:5]):  # 최대 5개의 QRS 피크만 시각화
        start = max(0, peak - window)
        end = min(len(signal), peak + window)
        segment = signal[start:end]

        # QRS 파형 시각화
        axs[0, i].plot(segment, label="QRS Wave", color="red")
        axs[0, i].axvline(window, color="blue", linestyle="--", label="QRS Peak")
        axs[0, i].set_title(f"QRS {i+1}")
        axs[0, i].legend()
        axs[0, i].grid()

    for i, peak in enumerate(p_peaks[:5]):  # 최대 5개의 P 피크만 시각화
        start = max(0, peak - window)
        end = min(len(signal), peak + window)
        segment = signal[start:end]

        # P 파형 시각화
        axs[1, i].plot(segment, label="P Wave", color="green")
        axs[1, i].axvline(window, color="blue", linestyle="--", label="P Peak")
        axs[1, i].set_title(f"P Wave {i+1}")
        axs[1, i].legend()
        axs[1, i].grid()

    plt.tight_layout()
    st.pyplot(fig)

# 예제: 기존 plot_ecg_with_peaks 함수에 추가
def plot_ecg_with_peaks(signal, fs=360):
    # 필터링
    filtered_signal = lowpass_filter(signal)

    # QRS 피크 검출
    qrs_peaks, _ = find_peaks(filtered_signal, distance=fs // 2, height=np.mean(filtered_signal))

    # P 파 검출
    p_peaks = detect_p_wave(filtered_signal, qrs_peaks, fs)

    # 전체 신호와 피크 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(signal, label="Original ECG Signal", alpha=0.5)
    ax.plot(filtered_signal, label="Filtered ECG Signal", color="blue")
    ax.plot(qrs_peaks, filtered_signal[qrs_peaks], "x", label="QRS Peaks", color="red")
    ax.plot(p_peaks, filtered_signal[p_peaks], "o", label="P Peaks", color="green")
    ax.legend()
    ax.set_title("ECG Signal with QRS and P Peaks")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # QRS 및 P 파형 확대 시각화
    plot_qrs_p_wave_shapes(filtered_signal, qrs_peaks, p_peaks, fs)


# Streamlit UI
st.title("부정맥 감지 시스템")

uploaded_files = st.file_uploader(
    "ECG 신호 파일 업로드 (.mat 또는 .hea 및 .dat 파일)",
    type=["mat", "hea", "dat"],
    accept_multiple_files=True,
)

if uploaded_files:
    try:
        # 파일 분류
        mat_file = None
        hea_file = None
        dat_file = None
        for file in uploaded_files:
            if file.name.endswith('.mat'):
                mat_file = file
            elif file.name.endswith('.hea'):
                hea_file = file
            elif file.name.endswith('.dat'):
                dat_file = file

        # 파일 처리
        if mat_file:
            signal = load_mat_file(mat_file)
        elif hea_file and dat_file:
            signal = load_hea_dat_files(hea_file, dat_file)
        else:
            st.error("필요한 파일(.hea 및 .dat)을 업로드하세요.")
            signal = None

        # 결과 확인 버튼 추가
        if st.button("결과 확인"):
            # 예측 실행
            if signal is not None:
                result, confidence = predict_signal(signal, model)
                
                # 결과 출력 (색상 및 크기 조정)
                color = "red" if result == "부정맥" else "blue"
                st.markdown(
                    f"""
                    <div style="text-align: center; margin-top: 20px;">
                        <h2 style="color: {color};">
                            예측 결과: {result} (확률: {confidence:.2f})
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                 
                
            else:
                st.error("신호 데이터를 불러올 수 없습니다.")
        # 특징 분석 및 판단 이유 확인 버튼
        if st.button("특징 확인 및 판단 이유 보기"):
            if signal is not None:
                try:
                    # 순시 주파수와 스펙트럼 엔트로피 계산
                    inst_freq, pentropy = calculate_features(signal)

                    # 특징 시각화
                    st.markdown("### 신호 및 주요 특징 시각화")
                    plot_features_with_heatmap(signal, inst_freq, pentropy)

                    # 모델 예측
                    result, confidence = predict_signal(signal, model)

                    # 특징 기반 판단 설명
                    st.markdown("### 모델의 판단 이유")
                    
                    if result == "부정맥":
                        st.markdown(
                            """
                            <div style="color: red;">
                                업로드된 심전도 신호는 모델의 순시주파수에서 나타난 비정상적인 주파수 변화와,\n 
                            <div style="color: red;">    
                                스펙트럼 엔트로피에서 확인된 높은 불규칙성을 기반으로 부정맥(심방세동)으로 판정되었습니다.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        # 추가된 주의사항 문구
                        st.markdown(
                            """
                            <div style="color: red; text-align: center; margin-top: 10px;">
                                <h4>⚠️ 주의사항:</h4>
                            <p>1. **스트레스를 피하세요**: 스트레스는 심장 건강에 큰 영향을 미칠 수 있습니다.</p>
                            <p>2. **카페인과 알코올 섭취를 줄이세요**: 카페인과 알코올은 부정맥 증상을 악화시킬 수 있습니다.</p>
                            <p>3. **규칙적인 운동**: 무리하지 않는 선에서 가벼운 운동을 지속하세요.</p>
                            <p>4. **약물 복용**: 처방받지 않은 약물을 복용하지 마세요. 기존에 복용 중인 약물은 의사와 상의 후 조정하세요.</p>
                            <p>5. **정기 검진**: 가까운 병원에서 정기적으로 심전도 검사를 받으세요.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    else:
                        st.markdown(
                            """
                            <div style="color: blue;">
                                업로드된 심전도 신호는 순시주파수에서 관찰된 규칙적인 주파수 변화와,\n 
                            <div style="color: blue;">    
                                스펙트럼 엔트로피에서 나타난 낮은 복잡도와 안정성을 기반으로 정상 신호로 판정되었습니다.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.error(f"특징 계산 중 오류 발생: {e}")
        # 파형 확인 버튼 추가
        if st.button("파형 확인 (QRS 및 P 파)"):
            if signal is not None:
                st.markdown("### QRS 및 P 파 시각화")
                plot_ecg_with_peaks(signal)  # QRS 및 P 파 시각화 함수 호출
            else:
                st.error("신호 데이터를 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

# 주의사항 추가
st.sidebar.info("⚠️ 이 애플리케이션은 의학적 진단을 위한 것이 아니며 연구용으로만 사용됩니다.") 