import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import fastapi as _fapi
import schemas as _schemas
import services as _services
from io import BytesIO
import base64
import traceback
import uvicorn
import threading
import numpy as np
from PIL import Image
import asyncio

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Aging Time Lapse API"}


# Endpoint to test the backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the Aging Time Lapse with FastAPI"}


@app.post("/api/aging/")
async def generate_image(imgPromptCreate: _schemas.ImageCreate = _fapi.Depends()):
    
    try:
        image_10, image_30, image_50, image_70 = await _services.generate_image(imgPrompt=imgPromptCreate)
    except Exception as e:
        print(traceback.format_exc())
        return {"message": f"{e.args}"}
    
    buffered = BytesIO()
    image_10.save(buffered, format="JPEG")
    encoded_img_10 = base64.b64encode(buffered.getvalue())
    
    buffered = BytesIO()
    image_30.save(buffered, format="JPEG")
    encoded_img_30 = base64.b64encode(buffered.getvalue())
    
    buffered = BytesIO()
    image_50.save(buffered, format="JPEG")
    encoded_img_50 = base64.b64encode(buffered.getvalue())
    
    buffered = BytesIO()
    image_70.save(buffered, format="JPEG")
    encoded_img_70 = base64.b64encode(buffered.getvalue())
    payload = {
        "mime" : "image/jpg",
        "image_10": encoded_img_10,
        "image_30": encoded_img_30,
        "image_50": encoded_img_50,
        "image_70": encoded_img_70
    }
    
    return payload

#gradio UI
def gradio_generate_image(image):
    """ 사용자가 업로드한 이미지를 Aging Time-lapse로 변환 """
    try:
        # Gradio에서 받은 이미지를 numpy 배열에서 PIL 이미지로 변환
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(np.array(image, dtype=np.uint8))  
        else:
            raise ValueError("Invalid image format received")

        # PIL 이미지를 FastAPI에서 기대하는 UploadFile로 변환
        from starlette.datastructures import UploadFile
        from io import BytesIO
        
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        upload_file = UploadFile(img_byte_arr, filename="uploaded.jpg")

        # 기존 서비스의 generate_image() 함수 호출 (비동기 실행)
        imgPromptCreate = _schemas.ImageCreate(encoded_base_img=upload_file)
        
        # `asyncio.run()` 사용하여 비동기 함수 실행
        image_10, image_30, image_50, image_70 = asyncio.run(_services.generate_image(imgPrompt=imgPromptCreate))

        return image_10, image_30, image_50, image_70

    except Exception as e:
        print(f"Error: {str(e)}")  # 로그 출력
        blank_image = Image.new('RGB', (512, 512), (255, 255, 255))  # 빈 흰색 이미지 생성
        return blank_image, blank_image, blank_image, blank_image  # 4개의 빈 이미지 반환


# Gradio 인터페이스 설정
gradio_interface = gr.Interface(
    fn=gradio_generate_image,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="pil"), gr.Image(type="pil"), gr.Image(type="pil"), gr.Image(type="pil")],
    title="Aging Time-Lapse Generator",
    description="Upload an image to generate an aging time-lapse (10, 30, 50, 70 years old).",
)

# Gradio를 별도의 스레드에서 실행
def run_gradio():
    gradio_interface.launch(server_name="0.0.0.0", server_port=7860, share=False)

@app.get("/gradio")
async def redirect_to_gradio():
    """ `/gradio`에 접속하면 Gradio 웹 UI로 리디렉션 """
    return RedirectResponse(url="http://127.0.0.1:7860")

# FastAPI 실행
if __name__ == "__main__":
    # Gradio UI를 별도의 스레드에서 실행
    threading.Thread(target=run_gradio, daemon=True).start()
    
    # FastAPI 실행 (workers=3 사용하면 Gradio가 동작하지 않음)
    uvicorn.run(app, host="0.0.0.0", port=8000)

