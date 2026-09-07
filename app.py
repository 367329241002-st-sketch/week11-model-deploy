"""
app.py
เว็บแอป Streamlit สำหรับจำแนกภาพ X-ray ปอด เป็น 3 กลุ่ม: covid / normal / pneumonia
โดยใช้โมเดลที่ฝึกไว้แล้วด้วยโปรแกรม Orange Data Mining (ไฟล์ .pkcls)

** ข้อมูลสำคัญที่ตรวจพบจากไฟล์โมเดลจริง (W10_treemodel / W10_svmmodel / W10_nnmodel) **
- โมเดลเหล่านี้ไม่ได้ฝึกจากตารางตัวเลข/ข้อความที่คนกรอกเองได้
- แต่ฝึกจาก "Image Embedding" ของ Orange: แต่ละภาพ X-ray จะถูกแปลงเป็นเวกเตอร์ตัวเลข
  2048 ค่า (คอลัมน์ n0 ... n2047) ด้วยโมเดล Inception v3 (ค่า default ของ Orange
  Image Embedding widget ซึ่งขนาด 2048 มิติตรงกับ Inception v3 พอดี)
- Class ที่ทำนาย 3 กลุ่ม: covid, normal, pneumonia
ดังนั้นแอปนี้จึงให้ผู้ใช้ "อัปโหลดภาพ X-ray" แทนการกรอกตัวเลขทีละช่อง แล้วแอปจะคำนวณ
embedding ให้อัตโนมัติก่อนส่งเข้าโมเดล

** ข้อกำหนดสำคัญตอนรัน **
- ขั้นตอนคำนวณ embedding ใช้ Orange3-ImageAnalytics ซึ่งเรียก API ฝั่งเซิร์ฟเวอร์ของ
  Orange (https://api.garaza.io/) ดังนั้นเครื่องที่รันแอปนี้ "ต้องต่ออินเทอร์เน็ตได้"
- ถ้าเครื่องที่รัน (เช่น Streamlit Community Cloud) เป็น Linux แบบไม่มีจอ (headless)
  อาจต้องติดตั้งไลบรารีระบบเพิ่มสำหรับ PyQt5 ดูหมายเหตุท้ายไฟล์ / ไฟล์ packages.txt
"""

import os
import tempfile

import joblib
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# 1) ตั้งค่าหน้าเว็บ และแสดงหัวข้อแอป
# ---------------------------------------------------------
st.set_page_config(page_title="โปรแกรมจำแนกโรค Covid จากภาพ X-ray", layout="centered")
st.title("โปรแกรมจำแนกโรค Covid  จากภาพ X-ray")

st.write(
    "อัปโหลดภาพ X-ray ปอด แล้วเลือกโมเดลที่ต้องการใช้ทำนายผล "
    "(covid / normal / pneumonia)"
)

# ---------------------------------------------------------
# 2) ฟังก์ชันช่วยโหลดโมเดลและตัวคำนวณ embedding
#    ใช้ st.cache_resource เพื่อไม่ต้องโหลดใหม่ทุกครั้งที่กดปุ่ม
# ---------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model(model_path_or_bytes, is_path: bool = True):
    """โหลดไฟล์โมเดล .pkcls ด้วย joblib"""
    if is_path:
        return joblib.load(model_path_or_bytes)
    return joblib.load(model_path_or_bytes)


@st.cache_resource(show_spinner=False)
def get_embedder(embedder_name: str):
    """
    สร้างตัวคำนวณ image embedding จาก orangecontrib.imageanalytics
    (import แบบ lazy ไว้ในฟังก์ชัน เพื่อให้แอปเริ่มทำงานได้แม้ยังไม่ได้ใช้ส่วนนี้)
    """
    from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

    return ImageEmbedder(model=embedder_name)


# ---------------------------------------------------------
# 3) ส่วนเลือกไฟล์โมเดล (.pkcls)
#    - เลือกจากโฟลเดอร์ models/ ในเครื่อง หรืออัปโหลดเอง
#    !! วางไฟล์ W10_treemodel.pkcls, W10_svmmodel.pkcls, W10_nnmodel.pkcls
#       ไว้ในโฟลเดอร์ models/ ก่อนรันแอป (หรือเลือกอัปโหลดแทน) !!
# ---------------------------------------------------------
st.header("1. เลือกไฟล์โมเดล")

MODEL_DIR = "models"
EXPECTED_MODEL_FILES = [
    "W10_treemodel.pkcls",   # Decision Tree
    "W10_svmmodel.pkcls",    # SVM
    "W10_nnmodel.pkcls",     # Neural Network
]

import glob  # noqa: E402  (import ตรงนี้เพื่อให้อ่านง่ายเป็นลำดับขั้นตอน)

model_files = []
if os.path.isdir(MODEL_DIR):
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkcls"))

model = None
model_source = st.radio(
    "เลือกวิธีการโหลดโมเดล",
    options=["เลือกจากโฟลเดอร์ในเครื่อง", "อัปโหลดไฟล์โมเดลเอง"],
)

if model_source == "เลือกจากโฟลเดอร์ในเครื่อง":
    if model_files:
        selected_model_name = st.selectbox(
            "เลือกไฟล์โมเดล (.pkcls)",
            options=model_files,
            format_func=lambda x: os.path.basename(x),
        )
        try:
            model = load_model(selected_model_name, is_path=True)
            st.success(f"โหลดโมเดล '{os.path.basename(selected_model_name)}' สำเร็จ")
        except Exception as e:
            st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    else:
        st.warning(
            f"ไม่พบไฟล์ .pkcls ในโฟลเดอร์ '{MODEL_DIR}/' "
            f"กรุณาวางไฟล์โมเดล ({', '.join(EXPECTED_MODEL_FILES)}) ไว้ในโฟลเดอร์นี้ "
            "หรือเลือกอัปโหลดไฟล์เองแทน"
        )
else:
    uploaded_model = st.file_uploader("อัปโหลดไฟล์โมเดล (.pkcls)", type=["pkcls", "pkl"])
    if uploaded_model is not None:
        try:
            model = load_model(uploaded_model, is_path=False)
            st.success("โหลดโมเดลจากไฟล์ที่อัปโหลดสำเร็จ")
        except Exception as e:
            st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")

# ---------------------------------------------------------
# 4) เลือก embedder ให้ตรงกับตอนฝึกโมเดล
#    ค่า default "inception-v3" ถูกเลือกไว้ให้ เพราะขนาด feature ของโมเดล
#    (2048 มิติ) ตรงกับ Inception v3 ซึ่งเป็นค่า default ของ Orange
#    Image Embedding widget พอดี — แต่ถ้าตอนฝึกโมเดลจริงมีการเปลี่ยน embedder
#    เป็นตัวอื่น ต้องเลือกให้ตรงกันที่นี่ ไม่เช่นนั้นผลทำนายจะผิดพลาด
# ---------------------------------------------------------
st.header("2. ตั้งค่า Image Embedding")

embedder_name = st.selectbox(
    "โมเดลที่ใช้แปลงภาพเป็นตัวเลข (Embedder) — ต้องตรงกับตอนฝึกโมเดล",
    options=["inception-v3", "painters", "vgg16", "vgg19", "squeezenet", "deeploc", "openface"],
    index=0,
    help="ค่า default 'inception-v3' เดาจากขนาด feature 2048 มิติของโมเดลที่อัปโหลดมา",
)

# ---------------------------------------------------------
# 5) อัปโหลดภาพ X-ray ที่ต้องการทำนาย
# ---------------------------------------------------------
st.header("3. อัปโหลดภาพ X-ray")

uploaded_image = st.file_uploader(
    "เลือกไฟล์ภาพ X-ray (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    st.image(uploaded_image, caption="ภาพที่อัปโหลด", use_container_width=True)

# ---------------------------------------------------------
# 6) ปุ่มทำนายผล
# ---------------------------------------------------------
st.header("4. ทำนายผล")

if st.button("ทำนายผล"):
    if model is None:
        st.error("กรุณาเลือก/อัปโหลดไฟล์โมเดลก่อน")
    elif uploaded_image is None:
        st.error("กรุณาอัปโหลดภาพ X-ray ก่อน")
    else:
        try:
            with st.spinner("กำลังคำนวณ embedding ของภาพ (ต้องใช้อินเทอร์เน็ต)..."):
                # -------------------------------------------------
                # บันทึกภาพที่อัปโหลดเป็นไฟล์ชั่วคราวบนดิสก์ก่อน
                # เพราะ ImageEmbedder ต้องการ "พาธไฟล์ภาพ" ไม่ใช่ bytes โดยตรง
                # -------------------------------------------------
                suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
                    tmp_f.write(uploaded_image.getbuffer())
                    tmp_path = tmp_f.name

                # -------------------------------------------------
                # คำนวณ embedding ด้วยโมเดลเดียวกับตอนฝึก (default: inception-v3)
                # ผลลัพธ์คือ list ของเวกเตอร์ตัวเลข (1 ภาพ = 1 เวกเตอร์ 2048 ค่า)
                # -------------------------------------------------
                embedder = get_embedder(embedder_name)
                embeddings = embedder([tmp_path])
                os.remove(tmp_path)

            if embeddings is None or embeddings[0] is None:
                st.error(
                    "คำนวณ embedding ไม่สำเร็จ (อาจเป็นเพราะไม่มีอินเทอร์เน็ต "
                    "หรือเซิร์ฟเวอร์ embedding ของ Orange ไม่ตอบสนอง)"
                )
            else:
                # -------------------------------------------------
                # จัดรูปแบบ embedding ให้เป็น numpy array ชนิด float64
                # ขนาด (1, จำนวน feature) ให้ตรงกับตอนฝึกโมเดล (Orange ต้องการ float64)
                # -------------------------------------------------
                X = np.array(embeddings, dtype=np.float64)

                n_expected = len(model.domain.attributes)
                if X.shape[1] != n_expected:
                    st.error(
                        f"ขนาด embedding ที่ได้ ({X.shape[1]} มิติ) ไม่ตรงกับที่โมเดลต้องการ "
                        f"({n_expected} มิติ) กรุณาเปลี่ยน Embedder ให้ตรงกับตอนฝึกโมเดล"
                    )
                else:
                    # -------------------------------------------------
                    # ส่งเข้าโมเดลเพื่อทำนายผล (โมเดล Orange เรียกแบบ model(X))
                    # -------------------------------------------------
                    pred_index = model(X)          # array ของ index class เช่น [2]
                    pred_proba = model(X, model.Probs)  # ความน่าจะเป็นของแต่ละ class

                    class_values = model.domain.class_var.values  # ('covid','normal','pneumonia')
                    predicted_label = class_values[int(pred_index[0])]
                    confidence = float(np.max(pred_proba[0])) * 100

                    # -------------------------------------------------
                    # แปลผลลัพธ์เป็นภาษาไทยให้อ่านง่าย
                    # -------------------------------------------------
                    label_map_th = {
                        "covid": "พบลักษณะเข้าข่ายโรค Covid-19",
                        "normal": "ปกติ ไม่พบความผิดปกติ",
                        "pneumonia": "พบลักษณะเข้าข่ายโรคปอดอักเสบ (Pneumonia)",
                    }
                    result_th = label_map_th.get(predicted_label, predicted_label)

                    st.success(f"ผลการทำนาย: {result_th} (ความมั่นใจ {confidence:.2f}%)")

                    # แสดงความน่าจะเป็นของทุก class เพื่อความโปร่งใส
                    st.write("ความน่าจะเป็นของแต่ละกลุ่ม:")
                    for cls_name, p in zip(class_values, pred_proba[0]):
                        st.write(f"- {cls_name}: {p * 100:.2f}%")

        except ModuleNotFoundError as e:
            st.error(
                "ยังไม่ได้ติดตั้งไลบรารีที่จำเป็นสำหรับคำนวณ embedding "
                f"({e}). กรุณาติดตั้งตาม requirements.txt ให้ครบ"
            )
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")

st.divider()
st.caption(
    "หมายเหตุ: การคำนวณ embedding ของภาพต้องเชื่อมต่ออินเทอร์เน็ตไปยังเซิร์ฟเวอร์ของ Orange "
    "(https://api.garaza.io/) ทุกครั้งที่ทำนายผล"
)
