# backend_simple.py - AI Medical Assistant ด้วย Gemini API + YOLO
# เวอร์ชันง่ายสำหรับนักเรียน

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image
import io
import os
from dotenv import load_dotenv
import pandas as pd
from typing import Optional
import numpy as np

# โหลด environment variables
load_dotenv()

# ตั้งค่า Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# สร้าง FastAPI app
app = FastAPI(title="AI Medical Assistant API")

# เปิดใช้ CORS เพื่อให้ Frontend เรียกใช้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ยอมรับทุก origin (สำหรับ development)
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด YOLO Models (Lazy Loading)
xray_model = None
blood_model = None

def get_xray_model():
    """โหลด YOLO model สำหรับ X-ray"""
    global xray_model
    if xray_model is None:
        print("🔄 กำลังโหลด X-ray model...")
        xray_model = YOLO("models/xray_best.pt")
        print("✅ โหลด X-ray model สำเร็จ!")
    return xray_model

def get_blood_model():
    """โหลด YOLO model สำหรับเลือด"""
    global blood_model
    if blood_model is None:
        print("🔄 กำลังโหลด Blood model...")
        blood_model = YOLO("models/blood_best.pt")
        print("✅ โหลด Blood model สำเร็จ!")
    return blood_model

# โหลดข้อมูลผู้ป่วย
try:
    patient_df = pd.read_csv("healthcare_dataset.csv")
    patient_df.columns = patient_df.columns.str.strip()
    print(f"✅ โหลดข้อมูลผู้ป่วย {len(patient_df)} รายการ")
except Exception as e:
    print(f"⚠️ ไม่สามารถโหลดข้อมูลผู้ป่วย: {e}")
    patient_df = None

# เก็บประวัติการสนทนาของแต่ละ session
chat_sessions = {}


# ==============================================
# 1. API หน้าแรก - ทดสอบว่า server ทำงาน
# ==============================================
@app.get("/")
def home():
    """API หน้าแรก ทดสอบว่า server ทำงาน"""
    return {
        "message": "🏥 AI Medical Assistant API พร้อมใช้งาน!",
        "powered_by": "Google Gemini",
        "version": "1.0"
    }


# ==============================================
# 2. API วิเคราะห์ภาพทางการแพทย์ด้วย YOLO
# ==============================================
@app.post("/analyze")
async def analyze_medical_image(
    file: UploadFile = File(...),
    image_type: str = Form(...)
):
    """
    วิเคราะห์ภาพทางการแพทย์ด้วย YOLO Models

    Parameters:
    - file: ไฟล์รูปภาพ (X-ray หรือเลือด)
    - image_type: ประเภท "xray" หรือ "blood"
    """
    try:
        # อ่านไฟล์ภาพ
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        if image_type == "xray":
            # ใช้ YOLO model วิเคราะห์ X-ray
            model = get_xray_model()
            results = model(image)

            # ดึงผลลัพธ์จาก YOLO
            if results[0].probs:  # Classification model
                probs = results[0].probs.data.cpu().numpy()
                classes = model.names

                best_idx = probs.argmax()
                best_class = classes[best_idx]
                confidence = float(probs[best_idx])

                return {
                    "success": True,
                    "type": "xray",
                    "result": best_class,
                    "confidence": f"{confidence:.1%}",
                    "all_predictions": {
                        classes[i]: f"{probs[i]:.1%}"
                        for i in range(len(probs))
                    },
                    "filename": file.filename
                }
            else:
                return {
                    "success": False,
                    "error": "Model ไม่สามารถวิเคราะห์ภาพได้"
                }

        elif image_type == "blood":
            # ใช้ YOLO model นับเซลล์เลือด
            model = get_blood_model()
            results = model(image)

            if results[0].boxes:  # Detection model
                boxes = results[0].boxes
                cell_counts = {}

                # นับจำนวนเซลล์แต่ละประเภท
                for box in boxes:
                    cls_id = int(box.cls)
                    class_name = model.names[cls_id]
                    cell_counts[class_name] = cell_counts.get(class_name, 0) + 1

                return {
                    "success": True,
                    "type": "blood",
                    "cell_counts": cell_counts,
                    "total_cells": len(boxes),
                    "filename": file.filename
                }
            else:
                return {
                    "success": True,
                    "type": "blood",
                    "cell_counts": {},
                    "total_cells": 0,
                    "message": "ไม่พบเซลล์เลือดในภาพ",
                    "filename": file.filename
                }

        else:
            return {
                "success": False,
                "error": "ประเภทภาพไม่ถูกต้อง กรุณาเลือก 'xray' หรือ 'blood'"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"เกิดข้อผิดพลาด: {str(e)}"
        }


# ==============================================
# 3. API แชทบอทให้คำปรึกษาทางการแพทย์
# ==============================================
@app.post("/chat")
async def chat_with_medical_ai(
    message: str = Form(...),
    session_id: str = Form(default="default")
):
    """
    แชทกับ AI แพทย์ให้คำปรึกษา

    Parameters:
    - message: ข้อความจากผู้ใช้
    - session_id: ID ของ session (สำหรับจดจำบทสนทนา)
    """
    try:
        # สร้าง session ใหม่ถ้ายังไม่มี
        if session_id not in chat_sessions:
            system_instruction = """คุณเป็น AI ผู้ช่วยแพทย์ที่เชี่ยวชาญ มีหน้าที่:

1. ตอบคำถามเกี่ยวกับสุขภาพและการแพทย์
2. อธิบายอาการและโรคต่างๆ แบบเข้าใจง่าย
3. ให้คำแนะนำการดูแลสุขภาพเบื้องต้น
4. แนะนำให้ปรึกษาแพทย์เสมอสำหรับการวินิจฉัยที่แม่นยำ

กฎสำคัญ:
- ตอบเป็นภาษาไทยที่เข้าใจง่าย
- ตอบสั้นกระชับ ไม่เกิน 100 คำ
- เป็นมิตรและให้กำลังใจ
- เตือนให้ปรึกษาแพทย์เสมอ
- ไม่วินิจฉัยโรคขาด แค่ให้ข้อมูลเบื้องต้น"""

            # กำหนดค่า generation config
            generation_config = genai.GenerationConfig(
                temperature=0.7,          # ความสร้างสรรค์ (0.0-2.0) ค่าต่ำ=ตรงไปตรงมา, สูง=สร้างสรรค์
                max_output_tokens=500,    # จำนวน token สูงสุดที่จะตอบกลับ
                top_p=0.95,               # Nucleus sampling (0.0-1.0)
                top_k=40,                 # จำนวนคำที่พิจารณา
            )

            model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                system_instruction=system_instruction,
                generation_config=generation_config
            )
            chat_sessions[session_id] = model.start_chat(history=[])

        # ดึง chat session
        chat = chat_sessions[session_id]

        # ส่งข้อความและรับคำตอบ (สามารถ override config ได้)
        response = chat.send_message(
            message,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
            )
        )

        return {
            "success": True,
            "response": response.text,
            "session_id": session_id
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"เกิดข้อผิดพลาด: {str(e)}"
        }


# ==============================================
# 4. API เคลียร์ประวัติการแชท
# ==============================================
@app.get("/chat/clear/{session_id}")
def clear_chat_session(session_id: str):
    """ลบประวัติการสนทนาของ session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"ลบประวัติการสนทนา {session_id} เรียบร้อย"}
    return {"message": "ไม่พบ session นี้"}


# ==============================================
# 5. API สถิติข้อมูลผู้ป่วย
# ==============================================
@app.get("/patient/stats")
async def get_patient_statistics():
    """ดึงสถิติข้อมูลผู้ป่วยจากฐานข้อมูล"""
    try:
        if patient_df is None:
            return {
                "success": False,
                "error": "ไม่สามารถโหลดข้อมูลผู้ป่วย"
            }

        # คำนวณสถิติต่างๆ
        stats = {
            "total_patients": int(len(patient_df)),
            "avg_age": float(patient_df['Age'].mean()),
            "conditions": patient_df['Medical Condition'].value_counts().to_dict(),
            "blood_types": patient_df['Blood Type'].value_counts().to_dict(),
            "gender": patient_df['Gender'].value_counts().to_dict(),
        }

        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"เกิดข้อผิดพลาด: {str(e)}"
        }


# ==============================================
# 6. API ค้นหาข้อมูลผู้ป่วย
# ==============================================
@app.post("/patient/search")
async def search_patient(
    search_query: str = Form(...),
    search_type: str = Form(default="name")
):
    """
    ค้นหาข้อมูลผู้ป่วย

    Parameters:
    - search_query: คำค้นหา
    - search_type: ประเภทการค้นหา (name, condition, blood_type)
    """
    try:
        if patient_df is None:
            return {
                "success": False,
                "error": "ไม่สามารถโหลดข้อมูลผู้ป่วย"
            }

        # ค้นหาตามประเภท
        if search_type == "name":
            results = patient_df[
                patient_df['Name'].str.contains(search_query, case=False, na=False)
            ]
        elif search_type == "condition":
            results = patient_df[
                patient_df['Medical Condition'].str.contains(search_query, case=False, na=False)
            ]
        elif search_type == "blood_type":
            results = patient_df[patient_df['Blood Type'] == search_query.upper()]
        else:
            results = patient_df[
                patient_df['Name'].str.contains(search_query, case=False, na=False)
            ]

        # จำกัดผลลัพธ์ 10 รายการ
        results = results.head(10)

        if len(results) == 0:
            return {
                "success": True,
                "message": "ไม่พบข้อมูลผู้ป่วย",
                "results": [],
                "total": 0
            }

        return {
            "success": True,
            "results": results.to_dict('records'),
            "total": len(results)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"เกิดข้อผิดพลาด: {str(e)}"
        }


# ==============================================
# รัน Server
# ==============================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 เริ่มต้น AI Medical Assistant Server...")
    print("📍 API จะทำงานที่: http://localhost:8000")
    print("📖 เอกสาร API: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
