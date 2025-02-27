import random
from gender_detection.gender_detection import predict_gender
import face_recognition


def set_seed():
    seed = random.randint(42,4294967295)
    return seed


#성별 예측
def gender(image):
    boxes_face = face_recognition.face_locations(image)
    if len(boxes_face) == 1:
        x0,y1,x1,y0 = boxes_face[0]
        face_image = image[x0:x1,y0:y1]
        
        gender_result = predict_gender(face_image)
        print(f"gender_result : {gender_result}")
        return gender_result
    else:
        print("*************")
        return 'Multiple faces', 1
