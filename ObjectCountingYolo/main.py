import cv2
import numpy as np
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import io
import base64
import uvicorn
app=FastAPI()
model=YOLO('yolo11n.pt')

@app.post("/detect")
async def detect_objects(file:UploadFile=File(...)):
    try:
        contents=await file.read()
        img_bytes=io.BytesIO(contents)
        img_arr=np.asarray(bytearray(img_bytes.read()),dtype=np.uint8)
        img=cv2.imdecode(img_arr,cv2.IMREAD_COLOR)

        results=model(img)
        cls_counts={}
        for result in results:
            boxes=result.boxes
            for box in boxes:
                cls_id=int(box.cls[0])
                cls_name=model.names[cls_id]
                conf=box.conf[0]
                if conf>0.5:
                    cls_counts[cls_name]=cls_counts.get(cls_name,0)+1
            ann_img=results[0].plot()
            _,buffer=cv2.imencode('.jpg',ann_img)
            ann_img_base_64=base64.b64encode(buffer).decode('utf-8')

            return JSONResponse(
                status_code=200,
                content={
                    "counts":cls_counts,
                    "annotated_image":ann_img_base_64
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message":f"Error :{str(e)}"}
        )
    
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)